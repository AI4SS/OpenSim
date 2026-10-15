"""Offline tests for semantic validation, request budgets and accepted caches."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

from opensim.evaluation import judging, semantic


def text(uid="requested_text_0123456789"):
    return {"text_uid": uid, "mode": "OpenSim", "event": "event", "side": "gen", "time_step": "t1", "text": uid}


def item(uid, *, rumor="spread"):
    labels = {dimension: values[0] for dimension, values in semantic.LABELS.items()}
    labels["rumor"] = rumor
    return {"text_id": uid, "labels": labels, "keywords": [], "is_invalid": False,
            "invalid_reason": "", "fluency_score": 4, "short_reason": "文本表达观点"}


def response(items):
    return {"content": json.dumps({"items": items}), "finish_reason": "stop", "refusal": None,
            "raw_response": {"items": items}}


class APIError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.status_code = code


class SemanticJudgingTests(unittest.TestCase):
    def judge(self, rows, request, *, attempts=None, strategies=None, max_attempts=5):
        calls, traces, accepted, sleeps = [], [], {}, []
        attempts = attempts if attempts is not None else {}
        def invoke(payload):
            calls.append(payload)
            return request(payload)
        judging.judge_batch(
            rows, model="gpt-4o-mini", labels=semantic.LABELS,
            prompt=lambda batch: semantic.build_prompt("event", {}, batch), parse=semantic.extract_json,
            request=invoke, accept=lambda row, result: accepted.update({row["text_uid"]: result}),
            evidence=lambda trace: traces.append(copy.deepcopy(trace)), attempts=attempts,
            strategies=strategies if strategies is not None else {}, max_attempts=max_attempts,
            backoff=2, max_backoff=3, sleep=sleeps.append,
        )
        return calls, traces, accepted, attempts, sleeps

    def test_invalid_then_legal_and_bounded_backoff(self):
        row = text()
        results = iter([response([item(row["text_uid"], rumor="neutral")]), response([item(row["text_uid"])])])
        calls, traces, accepted, attempts, sleeps = self.judge([row], lambda payload: next(results))
        self.assertEqual(attempts[row["text_uid"]], 2)
        self.assertEqual(accepted[row["text_uid"]]["labels"]["rumor"], "spread")
        self.assertEqual([call["response_format"]["type"] for call in calls], ["json_schema"] * 2)
        failed = next(trace for trace in traces if trace["status"] == "response_validated" and trace["invalid"])
        self.assertIn("Illegal or empty label", failed["invalid"][row["text_uid"]])
        self.assertEqual(sleeps, [2])

    def test_partial_batch_retries_only_missing_or_invalid(self):
        first, second = text("first_text_0123456789"), text("second_text_0123456789")
        results = iter([response([item(first["text_uid"])]), response([item(second["text_uid"])])])
        calls, _, accepted, attempts, _ = self.judge([first, second], lambda payload: next(results))
        self.assertEqual(set(accepted), {first["text_uid"], second["text_uid"]})
        ids = calls[1]["response_format"]["json_schema"]["schema"]["properties"]["items"]["items"]["properties"]["text_id"]["enum"]
        self.assertEqual(ids, [second["text_uid"]])
        self.assertEqual(attempts, {first["text_uid"]: 1, second["text_uid"]: 2})

    def test_exhaustion_and_resume_share_same_budget(self):
        row = text()
        bad = lambda payload: response([item(row["text_uid"], rumor="doubt")])
        calls, _, accepted, attempts, sleeps = self.judge([row], bad)
        self.assertEqual(len(calls), 5)
        self.assertEqual(accepted, {})
        self.assertEqual(sleeps, [2, 3, 3, 3])
        resumed, _, _, _, _ = self.judge([row], bad, attempts=attempts)
        self.assertEqual(resumed, [])

    def test_nonexact_and_duplicate_uids_rejected(self):
        row = text()
        for items in ([item(row["text_uid"][:12])], [item(row["text_uid"])] * 2, [item("unrequested")]):
            with self.subTest(items=items):
                _, _, accepted, _, _ = self.judge([row], lambda payload: response(items), max_attempts=1)
                self.assertEqual(accepted, {})

    def test_missing_metadata_and_empty_labels_rejected(self):
        row = text()
        invalid = item(row["text_uid"])
        invalid.pop("short_reason")
        self.assertEqual(self.judge([row], lambda payload: response([invalid]), max_attempts=1)[2], {})
        invalid = item(row["text_uid"], rumor="")
        self.assertEqual(self.judge([row], lambda payload: response([invalid]), max_attempts=1)[2], {})

    def test_explicit_unsupported_schema_fallback_uses_same_budget(self):
        row = text()
        modes = []
        def request(payload):
            modes.append(payload["response_format"]["type"])
            if modes[-1] == "json_schema":
                raise APIError(400, "response_format json_schema is not supported")
            return response([item(row["text_uid"])])
        _, traces, accepted, attempts, _ = self.judge([row], request)
        self.assertEqual(modes, ["json_schema", "json_object"])
        self.assertEqual(attempts[row["text_uid"]], 2)
        self.assertEqual(traces[1]["strategy_after"], "json_object")
        self.assertTrue(accepted)

    def test_auth_and_unrelated_bad_requests_are_not_retried(self):
        for code in (400, 401, 403, 404):
            request = Mock(side_effect=APIError(code, "invalid configuration"))
            with self.subTest(code=code), self.assertRaises(APIError):
                self.judge([text()], request)
            self.assertEqual(request.call_count, 1)

    def test_transient_failure_is_retried(self):
        row = text()
        responses = iter([APIError(429, "rate limited"), response([item(row["text_uid"])])])
        def request(payload):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value
        self.assertEqual(self.judge([row], request)[3][row["text_uid"]], 2)

    def test_schema_declares_every_enum_and_required_field(self):
        schema = judging.response_format(["uid"], semantic.LABELS)["json_schema"]
        self.assertTrue(schema["strict"])
        label_schema = schema["schema"]["properties"]["items"]["items"]["properties"]["labels"]
        self.assertFalse(label_schema["additionalProperties"])
        self.assertEqual(set(label_schema["required"]), set(semantic.DIMENSIONS))
        for dimension, values in semantic.LABELS.items():
            self.assertEqual(label_schema["properties"][dimension]["enum"], values)

    def test_sdk_has_no_nested_retries(self):
        constructor = Mock()
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=constructor)}), patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            state = threading.local()
            semantic.get_thread_openai_client(state)
            semantic.get_thread_openai_client(state)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)


class SemanticCacheTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(output_dir=str(root), model="gpt-4o-mini", mode="OpenSim", batch_size=1,
                               batch_workers_per_event=1, max_attempts=5, timeout=1, max_tokens=256,
                               backoff=0, max_backoff=0)

    def test_complete_cache_needs_no_client_and_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = text()
            path = root / "events/event_text_cache.jsonl"
            semantic.append_jsonl(path, semantic.build_text_record(row, item(row["text_uid"]), "gpt-4o-mini"))
            before = path.read_bytes()
            with patch.object(semantic, "get_thread_openai_client", side_effect=AssertionError("no client expected")), redirect_stdout(io.StringIO()):
                self.assertEqual(semantic.evaluate_event("event", [row], self.args(root), {})["missing"], 0)
            self.assertEqual(path.read_bytes(), before)

    def test_fatal_error_does_not_retry_other_pending_items_or_change_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = text("accepted")
            path = root / "events/event_text_cache.jsonl"
            semantic.append_jsonl(path, semantic.build_text_record(old, item(old["text_uid"]), "gpt-4o-mini"))
            before = path.read_bytes()
            create = Mock(side_effect=APIError(401, "auth failed"))
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            rows = [old] + [text(f"missing_{index}") for index in range(5)]
            with patch.object(semantic, "get_thread_openai_client", return_value=client), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "auth failed"):
                semantic.evaluate_event("event", rows, self.args(root), {})
            self.assertEqual(create.call_count, 1)
            self.assertEqual(path.read_bytes(), before)

    def test_attempt_log_counts_started_requests_once_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            for status in ("request_started", "request_failed"):
                semantic.append_jsonl(path, {"status": status, "text_uids": ["uid"], "attempts": {"uid": 1}, "strategy_after": "json_object"})
            self.assertEqual(semantic.load_attempts(path), ({"uid": 1}, {"uid": "json_object"}))
            path.write_text('{"truncated":')
            with self.assertRaisesRegex(ValueError, "Corrupt request evidence"):
                semantic.load_attempts(path)

    def test_conflicting_accepted_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            row = text()
            for label in ("spread", "counter"):
                semantic.append_jsonl(path, semantic.build_text_record(row, item(row["text_uid"], rumor=label), "gpt-4o-mini"))
            with self.assertRaisesRegex(ValueError, "Accepted cache record"):
                semantic.load_cache(path)

    def test_new_judgment_retries_invalid_then_appends_with_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = text("accepted"), text("new_text")
            path = root / "events/event_text_cache.jsonl"
            semantic.append_jsonl(path, semantic.build_text_record(old, item(old["text_uid"]), "gpt-4o-mini"))
            prefix = path.read_bytes()
            replies = iter([response([item(new["text_uid"], rumor="neutral")]), response([item(new["text_uid"], rumor="counter")])])
            def create(**payload):
                raw = next(replies)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw["content"], refusal=None), finish_reason="stop")], model_dump=lambda mode: raw["raw_response"])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=create))))
            with patch.object(semantic, "get_thread_openai_client", return_value=client), redirect_stdout(io.StringIO()):
                result = semantic.evaluate_event("event", [old, new], self.args(root), {})
            self.assertEqual(result["missing"], 0)
            self.assertEqual(client.chat.completions.create.call_count, 2)
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(semantic.load_cache(path)[new["text_uid"]]["rumor"], "counter")
            traces = [json.loads(line) for line in (root / "events/event_requests.jsonl").read_text().splitlines()]
            self.assertEqual(len([trace for trace in traces if trace["status"] == "response_received"]), 2)
            self.assertEqual(semantic.load_attempts(root / "events/event_requests.jsonl")[0], {new["text_uid"]: 2})

    def test_incomplete_cli_resume_does_not_reset_request_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairwise = root / "pairwise.csv"
            pairwise.write_text("event,time_step,agent_index,source_record_id,real_comment,generated_comment\nevent,t1,0,source,real,generated\n")
            output = root / "semantic"
            def create(**payload):
                ids = payload["response_format"]["json_schema"]["schema"]["properties"]["items"]["items"]["properties"]["text_id"]["enum"]
                raw = response([item(uid, rumor="neutral") for uid in ids])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw["content"], refusal=None), finish_reason="stop")], model_dump=lambda mode: raw["raw_response"])
            create_mock = Mock(side_effect=create)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
            argv = ["semantic", "--pairwise-csv", str(pairwise), "--output-dir", str(output), "--index-csv", str(root / "absent.csv"), "--max-attempts", "1"]
            with patch.object(sys, "argv", argv), patch.object(semantic, "get_thread_openai_client", return_value=client), redirect_stdout(io.StringIO()):
                self.assertEqual(semantic.main(), 2)
                evidence = output / "events/event_requests.jsonl"
                before = evidence.read_bytes()
                self.assertEqual(semantic.main(), 2)
                self.assertEqual(evidence.read_bytes(), before)
            self.assertEqual(create_mock.call_count, 1)

    def test_cli_cached_resume_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairwise = root / "pairwise.csv"
            pairwise.write_text("event,time_step,agent_index,source_record_id,real_comment,generated_comment\nevent,t1,0,source,real,generated\n")
            output = root / "semantic"
            _, rows = semantic.build_text_records("OpenSim", semantic.read_csv_rows(pairwise))
            cache = output / "events/event_text_cache.jsonl"
            for row in rows:
                semantic.append_jsonl(cache, semantic.build_text_record(row, item(row["text_uid"]), "gpt-4o-mini"))
            before = cache.read_bytes()
            argv = ["semantic", "--pairwise-csv", str(pairwise), "--output-dir", str(output), "--index-csv", str(root / "absent.csv")]
            with patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True), patch.object(semantic, "get_thread_openai_client", side_effect=AssertionError("no network")), redirect_stdout(io.StringIO()):
                self.assertEqual(semantic.main(), 0)
                self.assertEqual(semantic.main(), 0)
                pairwise.write_text(pairwise.read_text().replace("generated", "changed"))
                with self.assertRaisesRegex(ValueError, "inputs/model/prompt changed"):
                    semantic.main()
            self.assertEqual(cache.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
