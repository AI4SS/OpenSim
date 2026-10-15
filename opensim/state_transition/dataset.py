from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


STATE_KEYS = ("pos", "neu", "neg")
SOURCE_TYPES = ("comment", "share")
TEXT_HASH_DIM = 128
DAY_NUMERIC_KEYS = (
    "day_index_norm",
    "has_disturb",
    "disturb_count_log",
    "days_since_disturb_norm",
    "log_y_prev",
    "log_y_prev2",
    "log_y_prev_delta",
    "prev_comment_ratio",
    "prev_share_ratio",
    "prev_weighted_comment_ratio",
    "prev_weighted_share_ratio",
    "prev_daily_pos",
    "prev_daily_neu",
    "prev_daily_neg",
)
AGENT_NUMERIC_KEYS = (
    "followers_log",
    "friends_log",
    "statuses_log",
    "verified",
    "liked_log",
    "hot_log",
    "source_comment",
    "source_share",
)


POSITIVE_WORDS = (
    "支持", "赞", "好", "理性", "澄清", "感谢", "公正", "点赞", "相信", "真相",
    "加油", "优秀", "合理", "正能量", "期待", "认可", "官方", "通报", "证实",
    "辟谣", "清楚", "解释", "依法", "负责", "透明", "理解",
)
NEGATIVE_WORDS = (
    "质疑", "怒", "骂", "假", "黑", "离谱", "恶心", "可怜", "心痛", "严查",
    "不信", "造谣", "带节奏", "欺骗", "失望", "问题", "压力", "崩溃",
    "为什么", "怎么", "？", "?", "假的", "隐瞒", "反转", "打脸", "荒唐",
    "无语", "离谱", "投诉", "抵制", "曝光", "追责", "谎言", "洗白",
)
NEUTRAL_WORDS = (
    "关注", "看看", "蹲", "后续", "围观", "转发", "了解", "记录", "新闻",
    "情况", "目前", "当地", "相关", "等待", "发布",
)
SHARE_TEXTS = {"", "转发", "转发微博", "分享图片", "微博转发"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def log1p(value: Any) -> float:
    return math.log1p(max(safe_float(value), 0.0))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash_features(text: str, dim: int = TEXT_HASH_DIM) -> list[float]:
    vec = [0.0] * dim
    text = clean_text(text)
    if not text:
        return vec

    tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]{1,4}|\d+", text.lower())
    if not tokens:
        tokens = list(text[:256])
    for token in tokens[:512]:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[bucket] += sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class TextFeatureEncoder:
    def __init__(
        self,
        model_path: str | None = "Qwen/Qwen3-1.7B",
        cache_db: str | Path | None = None,
        device: str = "cpu",
        max_length: int = 128,
        batch_size: int = 32,
    ) -> None:
        self.model_path = clean_text(model_path)
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.conn: sqlite3.Connection | None = None
        self.dim = TEXT_HASH_DIM

        if not self.model_path:
            raise ValueError("The Qwen3 text encoder path must be configured.")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "hidden_size", TEXT_HASH_DIM))

        if cache_db is not None:
            cache_path = Path(cache_db)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(cache_path))
            self.conn.execute("create table if not exists text_embedding (key text primary key, vector text not null)")
            self.conn.commit()

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        texts = [clean_text(text) for text in texts]
        if not texts:
            return []

        results: list[list[float] | None] = []
        missing: list[tuple[int, str, str]] = []
        for idx, text in enumerate(texts):
            if not text:
                results.append([0.0] * self.dim)
                continue
            key = hashlib.sha1(f"{self.model_path}\0{self.max_length}\0{text}".encode("utf-8")).hexdigest()
            cached = None
            if self.conn is not None:
                row = self.conn.execute("select vector from text_embedding where key = ?", (key,)).fetchone()
                if row is not None:
                    cached = json.loads(row[0])
            results.append(cached)
            if cached is None:
                missing.append((idx, key, text))

        if missing:
            assert self.tokenizer is not None
            assert self.model is not None
            rows_to_write = []
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [item[2] for item in chunk],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    hidden = outputs.last_hidden_state.float()
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1).cpu().tolist()
                for (idx, key, _), vec in zip(chunk, vecs):
                    results[idx] = vec
                    rows_to_write.append((key, json.dumps(vec, ensure_ascii=False)))
            if self.conn is not None and rows_to_write:
                self.conn.executemany(
                    "insert or replace into text_embedding(key, vector) values (?, ?)",
                    rows_to_write,
                )
                self.conn.commit()

        return [result if result is not None else [0.0] * self.dim for result in results]


def dist_from_dict(value: dict[str, Any] | None, default: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> list[float]:
    if not isinstance(value, dict):
        return list(default)
    vals = [max(safe_float(value.get(key)), 0.0) for key in STATE_KEYS]
    total = sum(vals)
    if total <= 0:
        return list(default)
    return [v / total for v in vals]


def mix_dist(primary: list[float], prior: list[float], primary_weight: float) -> list[float]:
    primary_weight = min(max(primary_weight, 0.0), 1.0)
    vals = [
        primary_weight * primary[i] + (1.0 - primary_weight) * prior[i]
        for i in range(3)
    ]
    total = sum(vals)
    if total <= 0:
        return [0.0, 1.0, 0.0]
    return [value / total for value in vals]


def extract_t_index(record: dict[str, Any]) -> int:
    value = record.get("t") or record.get("source_t") or record.get("day") or record.get("day_index")
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return int(safe_float(value, 1.0))


def _coerce_time_index(value: Any) -> int | None:
    """Parse a one-based source time without inventing a fallback value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        parsed = safe_float(value, float("nan"))
        if math.isfinite(parsed):
            return int(parsed)
        return None
    match = re.search(r"\d+", clean_text(value))
    return int(match.group(0)) if match else None


def macro_source_t_row_positions(
    macro_rows: list[dict[str, Any]],
) -> dict[int, int] | None:
    """Return the explicit source-time to sequence-row binding when present.

    Cleaned macro companions retain non-contiguous source times.  A complete,
    unique ``diagnostics.source_t`` column is therefore authoritative.  Older
    macro files do not carry that flag; returning ``None`` keeps their existing
    row-order behavior instead of guessing a partial mapping.
    """
    source_times: list[int] = []
    for row in macro_rows:
        diagnostics = row.get("diagnostics") or {}
        if not isinstance(diagnostics, dict) or "source_t" not in diagnostics:
            return None
        source_t = _coerce_time_index(diagnostics.get("source_t"))
        if source_t is None:
            return None
        source_times.append(source_t)
    if len(set(source_times)) != len(source_times):
        return None
    return {source_t: row_position for row_position, source_t in enumerate(source_times)}


def record_row_position(
    record: dict[str, Any],
    source_t_positions: dict[int, int] | None = None,
) -> int | None:
    """Map a micro record to a macro sequence row.

    With an explicit macro binding, an unmapped source time is skipped rather
    than being coerced to a neighboring row.  Without one, retain the legacy
    ``t - 1`` convention.
    """
    value = record.get("t")
    if value is None:
        value = record.get("source_t")
    if value is None:
        value = record.get("day")
    if value is None:
        value = record.get("day_index")
    source_t = _coerce_time_index(value)
    if source_t is None:
        return None
    if source_t_positions is not None:
        return source_t_positions.get(source_t)
    return source_t - 1


def sample_records_by_day(
    event_id: str,
    micro_rows: list[dict[str, Any]],
    agents_per_day: int,
    seed: int,
    macro_rows: list[dict[str, Any]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Sample comment/share agents using the ST dataset's exact ordering."""
    source_t_positions = (
        macro_source_t_row_positions(macro_rows) if macro_rows is not None else None
    )
    by_day: dict[int, list[dict[str, Any]]] = {}
    for record in micro_rows:
        if content_type(record) == "post":
            continue
        row_position = record_row_position(record, source_t_positions)
        if row_position is None:
            continue
        by_day.setdefault(row_position, []).append(record)

    sampled: dict[int, list[dict[str, Any]]] = {}
    for row_position, records in by_day.items():
        rows = list(records)
        rng_seed = seed + int(
            hashlib.md5(f"{event_id}:{row_position}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        rng = random.Random(rng_seed)
        if len(rows) > agents_per_day:
            rows = rng.sample(rows, agents_per_day)
        sampled[row_position] = rows[:agents_per_day]
    return sampled


def target_mask_from_diagnostics(
    diagnostics: dict[str, Any] | None,
    field_name: str,
    default: float = 1.0,
) -> float:
    """Read an explicit binary target mask, defaulting for legacy rows."""
    if not isinstance(diagnostics, dict) or field_name not in diagnostics:
        return default
    return 1.0 if safe_float(diagnostics.get(field_name), default) > 0.0 else 0.0


def cumulative_update_weight_from_diagnostics(
    diagnostics: dict[str, Any] | None,
    row: dict[str, Any],
) -> float:
    """Use retained cumulative weights when explicitly supplied.

    Legacy files did not expose this field and retain the historical fallback.
    In the cleaned format, an explicit zero is meaningful and must not be
    replaced by sample or post metrics.
    """
    if isinstance(diagnostics, dict) and diagnostics.get("cumulative_update_weight") is not None:
        return max(safe_float(diagnostics.get("cumulative_update_weight")), 0.0)
    return max(
        safe_float((diagnostics or {}).get("cumulative_update_weight")),
        safe_float((diagnostics or {}).get("sample_num_interactors")),
        safe_float((row.get("output") or {}).get("num_interactions")),
        1.0,
    )


def content_type(record: dict[str, Any]) -> str:
    value = clean_text(record.get("content_type")).lower()
    if value in {"share", "repost", "retweet"}:
        return "share"
    if value == "post":
        return "post"
    return "comment"


def record_text(record: dict[str, Any]) -> str:
    return clean_text(record.get("text") or record.get("content") or record.get("original_text"))


def profile_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("profile_text"),
        record.get("user_description"),
        record.get("verified_reason"),
        record.get("screen_name") or record.get("username") or record.get("nickname"),
        record.get("ip_location") or record.get("user_location"),
        record.get("gender"),
    ]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def weak_label(record: dict[str, Any], day_target: list[float]) -> tuple[list[float], float]:
    text = record_text(record)
    source = content_type(record)
    if source == "post":
        return [0.0, 1.0, 0.0], 0.0

    if text in SHARE_TEXTS:
        # Pure shares carry weak stance evidence; let macro supervision dominate.
        return list(day_target), 0.10

    pos = sum(1 for word in POSITIVE_WORDS if word in text)
    neg = sum(1 for word in NEGATIVE_WORDS if word in text)
    neu = sum(1 for word in NEUTRAL_WORDS if word in text)
    if "不" in text and any(word in text for word in ("相信", "支持", "公开", "公布")):
        neg += 1
    if re.search(r"[!！]{2,}", text):
        neg += 1
    if "求证" in text or "真假" in text or "真的假的" in text:
        neg += 1

    base_conf = 0.38 if source == "share" else 0.62
    margin = abs(pos - neg)
    evidence = min(0.25, 0.07 * (pos + neg + neu) + 0.04 * margin)

    if pos > neg and pos > 0:
        confidence = min(0.92, base_conf + evidence)
        label = [0.84, 0.12, 0.04] if margin >= 2 else [0.72, 0.22, 0.06]
    elif neg > pos and neg > 0:
        confidence = min(0.92, base_conf + evidence)
        label = [0.04, 0.12, 0.84] if margin >= 2 else [0.06, 0.22, 0.72]
    elif pos > 0 and neg > 0:
        confidence = min(0.78, base_conf + evidence)
        label = [0.42, 0.16, 0.42]
    elif neu > 0:
        confidence = min(0.62, 0.34 + evidence)
        label = [0.12, 0.76, 0.12]
    elif len(text) <= 3:
        confidence = 0.32 if source == "comment" else 0.16
        label = [0.1, 0.8, 0.1]
    else:
        confidence = 0.42 if source == "comment" else 0.20
        label = [0.18, 0.64, 0.18]

    return mix_dist(label, day_target, confidence), confidence


def day_numeric_features(rows: list[dict[str, Any]], day_idx: int) -> list[float]:
    row = rows[day_idx]
    prev = rows[day_idx - 1] if day_idx > 0 else None
    inp = row.get("input") or {}
    prev_diag = (prev or {}).get("diagnostics") or {}
    prev_output = (prev or {}).get("output") or {}
    prev2_output = (rows[day_idx - 2].get("output") if day_idx > 1 else {}) or {}
    prev_ratios = prev_diag.get("comment_share_ratio") or {}
    prev_weighted = prev_diag.get("weighted_interactions") or {}
    prev_daily = dist_from_dict((prev or {}).get("daily_state_distribution"), default=(0.0, 1.0, 0.0))
    max_day = max(len(rows) - 1, 1)
    days_since = inp.get("days_since_disturb")
    days_since_value = 30.0 if days_since is None else min(safe_float(days_since), 30.0)
    y_prev = safe_float(prev_output.get("num_interactions"))
    y_prev2 = safe_float(prev2_output.get("num_interactions"))
    return [
        safe_float(row.get("day_index"), day_idx) / max_day,
        1.0 if inp.get("has_disturb") else 0.0,
        log1p(inp.get("disturb_count")),
        days_since_value / 30.0,
        log1p(y_prev),
        log1p(y_prev2),
        log1p(y_prev) - log1p(y_prev2),
        safe_float(prev_ratios.get("comment")),
        safe_float(prev_ratios.get("share")),
        safe_float(prev_weighted.get("comment_ratio")),
        safe_float(prev_weighted.get("share_ratio")),
        *prev_daily,
    ]


def agent_numeric_features(record: dict[str, Any]) -> list[float]:
    verified = record.get("verified")
    verified_value = 1.0 if verified in {True, "true", "True", "1", 1} else 0.0
    return [
        log1p(record.get("followers_count")),
        log1p(record.get("friends_count") or record.get("following_count")),
        log1p(record.get("statuses_count") or record.get("user_post_count")),
        verified_value,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


@dataclass(frozen=True)
class LatentEventDims:
    day_dim: int
    agent_dim: int


class LatentStateEventDataset(Dataset):
    def __init__(
        self,
        macro_dir: str | Path,
        micro_dir: str | Path,
        event_ids: list[str] | None = None,
        agents_per_day: int = 128,
        k_steps: int = 3,
        seed: int = 42,
        text_encoder_path: str | None = "Qwen/Qwen3-1.7B",
        text_cache_db: str | Path | None = None,
        text_encoder_device: str = "cpu",
        text_max_length: int = 128,
        text_batch_size: int = 32,
        semantic_agent_profiles: bool = False,
        text_feature_encoder: TextFeatureEncoder | None = None,
    ) -> None:
        self.macro_dir = Path(macro_dir)
        self.micro_dir = Path(micro_dir)
        self.agents_per_day = agents_per_day
        self.k_steps = k_steps
        self.seed = seed
        self.day_text_features = text_feature_encoder or TextFeatureEncoder(
            model_path=text_encoder_path,
            cache_db=text_cache_db,
            device=text_encoder_device,
            max_length=text_max_length,
            batch_size=text_batch_size,
        )
        self.agent_text_features = (
            self.day_text_features
            if semantic_agent_profiles
            else TextFeatureEncoder()
        )

        macro_paths = sorted(self.macro_dir.glob("*.jsonl"))
        if event_ids is not None:
            wanted = set(event_ids)
            macro_paths = [path for path in macro_paths if path.stem in wanted]
        self.events: list[tuple[str, Path, Path]] = []
        for macro_path in macro_paths:
            micro_path = self.micro_dir / f"{macro_path.stem}.json"
            if micro_path.exists():
                self.events.append((macro_path.stem, macro_path, micro_path))
        if not self.events:
            raise ValueError(f"No paired macro/micro files found under {self.macro_dir} and {self.micro_dir}")

        self.dims = LatentEventDims(
            day_dim=3 + self.day_text_features.dim * 2 + len(DAY_NUMERIC_KEYS),
            agent_dim=self.agent_text_features.dim + len(AGENT_NUMERIC_KEYS),
        )

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        event_id, macro_path, micro_path = self.events[idx]
        macro_rows = [json.loads(line) for line in macro_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        micro_rows = json.loads(micro_path.read_text(encoding="utf-8"))

        T = len(macro_rows)
        A = self.agents_per_day
        day_features = torch.zeros(T, self.dims.day_dim, dtype=torch.float32)
        agent_features = torch.zeros(T, A, self.dims.agent_dim, dtype=torch.float32)
        agent_mask = torch.zeros(T, A, dtype=torch.float32)
        weak_labels = torch.zeros(T, A, 3, dtype=torch.float32)
        weak_conf = torch.zeros(T, A, dtype=torch.float32)
        daily_gt = torch.zeros(T, 3, dtype=torch.float32)
        cumulative_gt = torch.zeros(T, 3, dtype=torch.float32)
        future_daily_gt = torch.zeros(T, self.k_steps, 3, dtype=torch.float32)
        future_cumulative_gt = torch.zeros(T, self.k_steps, 3, dtype=torch.float32)
        future_daily_delta_gt = torch.zeros(T, self.k_steps, 3, dtype=torch.float32)
        future_cumulative_delta_gt = torch.zeros(T, self.k_steps, 3, dtype=torch.float32)
        future_impact_gt = torch.zeros(T, self.k_steps, dtype=torch.float32)
        # Keep the legacy boundary-only mask; target-specific masks below add
        # declared daily/cumulative validity without masking sequence context.
        future_mask = torch.zeros(T, self.k_steps, dtype=torch.float32)
        future_daily_mask = torch.zeros(T, self.k_steps, dtype=torch.float32)
        future_cumulative_mask = torch.zeros(T, self.k_steps, dtype=torch.float32)
        daily_target_mask = torch.ones(T, dtype=torch.float32)
        cumulative_target_mask = torch.ones(T, dtype=torch.float32)
        update_weight = torch.ones(T, dtype=torch.float32)
        sampled_records = sample_records_by_day(
            event_id,
            micro_rows,
            A,
            self.seed,
            macro_rows=macro_rows,
        )
        # Read declared target masks for every macro row before constructing
        # any future horizon.  Future rows are encountered later in the main
        # sequence loop, so assigning them there would silently leave all
        # future masks at their initialization value.
        for target_idx, target_row in enumerate(macro_rows):
            target_diagnostics = target_row.get("diagnostics") or {}
            daily_target_mask[target_idx] = target_mask_from_diagnostics(
                target_diagnostics,
                "daily_target_mask",
            )
            cumulative_target_mask[target_idx] = target_mask_from_diagnostics(
                target_diagnostics,
                "cumulative_target_mask",
            )

        for t, row in enumerate(macro_rows):
            inp = row.get("input") or {}
            diag = row.get("diagnostics") or {}
            prev_mu = (
                dist_from_dict(macro_rows[t - 1].get("cumulative_state_distribution"))
                if t > 0
                else [0.0, 1.0, 0.0]
            )
            event_text = clean_text(inp.get("event"))
            disturb_text = " ".join(clean_text(x) for x in inp.get("disturb_names") or [])
            prev_mf = clean_text(inp.get("previous_mf_summary"))
            day_vec = (
                prev_mu
                + self.day_text_features.encode(event_text + " " + disturb_text)
                + self.day_text_features.encode(prev_mf)
                + day_numeric_features(macro_rows, t)
            )
            day_features[t] = torch.tensor(day_vec, dtype=torch.float32)
            daily = dist_from_dict(row.get("daily_state_distribution"))
            cumulative = dist_from_dict(row.get("cumulative_state_distribution"))
            daily_gt[t] = torch.tensor(daily, dtype=torch.float32)
            cumulative_gt[t] = torch.tensor(cumulative, dtype=torch.float32)
            update_weight[t] = cumulative_update_weight_from_diagnostics(diag, row)

            records = list(sampled_records.get(t, []))
            if not records:
                agent_mask[t, 0] = 1.0
                weak_labels[t, 0] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

            records = records[:A]
            profile_vecs = self.agent_text_features.encode_many([profile_text(record) for record in records])
            for a_idx, record in enumerate(records):
                agent_vec = profile_vecs[a_idx] + agent_numeric_features(record)
                label, confidence = weak_label(record, daily)
                agent_features[t, a_idx] = torch.tensor(agent_vec, dtype=torch.float32)
                agent_mask[t, a_idx] = 1.0
                weak_labels[t, a_idx] = torch.tensor(label, dtype=torch.float32)
                weak_conf[t, a_idx] = float(confidence)

            for k in range(self.k_steps):
                target_idx = t + k + 1
                if target_idx < T:
                    future_daily = dist_from_dict(macro_rows[target_idx].get("daily_state_distribution"))
                    future_cumulative = dist_from_dict(macro_rows[target_idx].get("cumulative_state_distribution"))
                    daily_delta = [future_daily[i] - daily[i] for i in range(3)]
                    cumulative_delta = [future_cumulative[i] - cumulative[i] for i in range(3)]
                    future_daily_gt[t, k] = torch.tensor(future_daily, dtype=torch.float32)
                    future_cumulative_gt[t, k] = torch.tensor(future_cumulative, dtype=torch.float32)
                    future_daily_delta_gt[t, k] = torch.tensor(daily_delta, dtype=torch.float32)
                    future_cumulative_delta_gt[t, k] = torch.tensor(cumulative_delta, dtype=torch.float32)
                    future_impact_gt[t, k] = float(
                        sum(abs(value) for value in daily_delta)
                        + 0.5 * sum(abs(value) for value in cumulative_delta)
                    )
                    future_mask[t, k] = 1.0
                    future_daily_mask[t, k] = daily_target_mask[target_idx]
                    future_cumulative_mask[t, k] = cumulative_target_mask[target_idx]

        return {
            "event_id": event_id,
            "day_features": day_features,
            "agent_features": agent_features,
            "agent_mask": agent_mask,
            "weak_labels": weak_labels,
            "weak_conf": weak_conf,
            "daily_gt": daily_gt,
            "cumulative_gt": cumulative_gt,
            "future_daily_gt": future_daily_gt,
            "future_cumulative_gt": future_cumulative_gt,
            "future_daily_delta_gt": future_daily_delta_gt,
            "future_cumulative_delta_gt": future_cumulative_delta_gt,
            "future_impact_gt": future_impact_gt,
            "future_mask": future_mask,
            "future_daily_mask": future_daily_mask,
            "future_cumulative_mask": future_cumulative_mask,
            "daily_target_mask": daily_target_mask,
            "cumulative_target_mask": cumulative_target_mask,
            "update_weight": update_weight,
            "seq_len": T,
        }


def collate_latent_events(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    T_max = max(item["seq_len"] for item in batch)
    A = batch[0]["agent_features"].shape[1]
    K = batch[0]["future_daily_gt"].shape[1]
    day_dim = batch[0]["day_features"].shape[-1]
    agent_dim = batch[0]["agent_features"].shape[-1]

    out = {
        "event_id": [item["event_id"] for item in batch],
        "day_features": torch.zeros(B, T_max, day_dim, dtype=torch.float32),
        "agent_features": torch.zeros(B, T_max, A, agent_dim, dtype=torch.float32),
        "agent_mask": torch.zeros(B, T_max, A, dtype=torch.float32),
        "weak_labels": torch.zeros(B, T_max, A, 3, dtype=torch.float32),
        "weak_conf": torch.zeros(B, T_max, A, dtype=torch.float32),
        "daily_gt": torch.zeros(B, T_max, 3, dtype=torch.float32),
        "cumulative_gt": torch.zeros(B, T_max, 3, dtype=torch.float32),
        "future_daily_gt": torch.zeros(B, T_max, K, 3, dtype=torch.float32),
        "future_cumulative_gt": torch.zeros(B, T_max, K, 3, dtype=torch.float32),
        "future_daily_delta_gt": torch.zeros(B, T_max, K, 3, dtype=torch.float32),
        "future_cumulative_delta_gt": torch.zeros(B, T_max, K, 3, dtype=torch.float32),
        "future_impact_gt": torch.zeros(B, T_max, K, dtype=torch.float32),
        "future_mask": torch.zeros(B, T_max, K, dtype=torch.float32),
        "future_daily_mask": torch.zeros(B, T_max, K, dtype=torch.float32),
        "future_cumulative_mask": torch.zeros(B, T_max, K, dtype=torch.float32),
        "daily_target_mask": torch.zeros(B, T_max, dtype=torch.float32),
        "cumulative_target_mask": torch.zeros(B, T_max, dtype=torch.float32),
        "update_weight": torch.ones(B, T_max, dtype=torch.float32),
        "day_mask": torch.zeros(B, T_max, dtype=torch.float32),
        "seq_len": torch.tensor([item["seq_len"] for item in batch], dtype=torch.long),
    }

    for i, item in enumerate(batch):
        T = item["seq_len"]
        for key in (
            "day_features",
            "agent_features",
            "agent_mask",
            "weak_labels",
            "weak_conf",
            "daily_gt",
            "cumulative_gt",
            "future_daily_gt",
            "future_cumulative_gt",
            "future_daily_delta_gt",
            "future_cumulative_delta_gt",
            "future_impact_gt",
            "future_mask",
            "future_daily_mask",
            "future_cumulative_mask",
            "daily_target_mask",
            "cumulative_target_mask",
            "update_weight",
        ):
            out[key][i, :T] = item[key]
        out["day_mask"][i, :T] = 1.0
    return out
