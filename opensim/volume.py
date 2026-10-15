#!/usr/bin/env python3
"""Train-only causal autoregressive interaction-volume forecaster.

The runtime feature at day t contains only static event metadata, the
preregistered exogenous intervention schedule at t, and interaction counts
realized before t.  It deliberately excludes current/future engagement,
labels, states, posts and text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORECASTER_TYPE = "causal_autoregressive"
VARIANT = "causal_autoregressive"
TARGET_KEY = "sample_num_interactors"
CANONICAL_DISTURB_NAMES = {
    "宣传": "推广", "推广": "推广", "回应": "回应", "辟谣": "辟谣",
    "通报": "通报", "限流": "限流", "对抗": "对抗",
    "机器人": "机器人", "发布": "发布",
}
DEFAULT_EXCLUDED_EVENT_IDS = (
    "event_response_jiangping",
)
FORBIDDEN_FEATURE_TOKENS = (
    "natural", "current_state", "state_distribution", "label", "target",
    "likes", "comments_count", "shared_count", "share_ratio", "text",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_paths(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.jsonl"))


def event_parts(path_or_name: str | Path) -> tuple[str, str, str]:
    stem = Path(path_or_name).stem.replace("_sampler_gt", "")
    parts = stem.split("_")
    event_type = parts[0] if parts else "unknown"
    intervention = parts[1] if len(parts) > 1 else "unknown"
    return event_type, intervention, stem


def raw_disturb_names(row: Mapping[str, Any]) -> list[str]:
    value = row.get("input") or {}
    names = value.get("disturb_names") if isinstance(value, Mapping) else []
    return [CANONICAL_DISTURB_NAMES.get(str(name), str(name)) for name in (names or [])]


def disturb_type(row: Mapping[str, Any]) -> str:
    names = [name for name in raw_disturb_names(row) if name != "发布"]
    return "+".join(sorted(set(names))) if names else "none"


def observed_target(row: Mapping[str, Any]) -> float:
    diagnostics = row.get("diagnostics") or {}
    if isinstance(diagnostics, Mapping):
        value = safe_float(diagnostics.get(TARGET_KEY), -1.0)
        if value >= 0:
            return value
    output = row.get("output") or {}
    if isinstance(output, Mapping):
        return max(0.0, safe_float(output.get("num_interactions"), 0.0))
    return 0.0


def _phase(days_since: float) -> str:
    if days_since >= 30:
        return "none"
    if days_since <= 0:
        return "d0"
    if days_since == 1:
        return "d1"
    if days_since == 2:
        return "d2"
    return "d3plus"


def row_feature(
    path_or_name: str | Path,
    rows: Sequence[Mapping[str, Any]],
    day_idx: int,
    *,
    realized_history: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build the leakage-safe feature for day ``day_idx``.

    ``realized_history`` may contain only days strictly before ``day_idx``.
    At training time it is the previous train targets; at runtime it is the
    recursive simulated writeback.
    """
    if day_idx < 0 or day_idx >= len(rows):
        raise IndexError(day_idx)
    event_type, intervention, _event_id = event_parts(path_or_name)
    row = rows[day_idx]
    inp = row.get("input") or {}
    if not isinstance(inp, Mapping):
        inp = {}
    if realized_history is None:
        history = [observed_target(item) for item in rows[:day_idx]]
    else:
        if len(realized_history) != day_idx:
            raise ValueError("realized_history must contain exactly prior days")
        history = [max(0.0, safe_float(value)) for value in realized_history]
    prev1 = history[-1] if len(history) >= 1 else 0.0
    prev2 = history[-2] if len(history) >= 2 else 0.0
    prev3 = history[-3] if len(history) >= 3 else 0.0
    window = history[-3:]
    previous_mean = sum(window) / len(window) if window else 0.0
    previous_max = max(window) if window else 0.0
    days_since = min(max(safe_float(inp.get("days_since_disturb"), 30.0), 0.0), 30.0)
    names = [name for name in raw_disturb_names(row) if name != "发布"]
    total_days = max(len(rows), 1)
    feature = {
        "event_type": event_type,
        "event_intervention_family": intervention,
        "scheduled_disturb_type": "+".join(sorted(set(names))) if names else "none",
        "scheduled_primary_intervention": names[0] if names else "none",
        "scheduled_intervention_phase": _phase(days_since),
        "day_index": float(day_idx + 1),
        "day_index_norm": float(day_idx + 1) / total_days,
        "remaining_day_norm": float(total_days - day_idx - 1) / total_days,
        "is_first_day": float(day_idx == 0),
        "is_early_day_3": float(day_idx < 3),
        "scheduled_has_disturb": float(bool(names) or bool(inp.get("has_disturb"))),
        "scheduled_disturb_count": math.log1p(max(0.0, safe_float(inp.get("disturb_count"), len(names)))),
        "scheduled_days_since_disturb": days_since / 30.0,
        "scheduled_time_decay": math.exp(-0.5 * days_since) if days_since < 30 else 0.0,
        "log_y_prev1": math.log1p(prev1),
        "log_y_prev2": math.log1p(prev2),
        "log_y_prev3": math.log1p(prev3),
        "log_y_prev_delta": math.log1p(prev1) - math.log1p(prev2),
        "log_y_prev_mean_3": math.log1p(previous_mean),
        "log_y_prev_max_3": math.log1p(previous_max),
        "log_cumulative_y_prev": math.log1p(sum(history)),
        "prior_days_available": float(len(history)),
    }
    lowered = " ".join(feature).lower()
    if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
        raise AssertionError("forbidden current/future field entered causal feature schema")
    return feature


def feature_schema() -> list[str]:
    dummy = [{"input": {"disturb_names": [], "days_since_disturb": 30}}]
    return sorted(row_feature("event_none_dummy.jsonl", dummy, 0))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "sha256": file_sha256(path), "rows": len(load_jsonl(path))}
        for path in paths
    ]


def build_training_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    features: list[dict[str, Any]] = []
    targets: list[float] = []
    groups: list[str] = []
    for path in paths:
        rows = load_jsonl(path)
        history: list[float] = []
        for day_idx, row in enumerate(rows):
            features.append(row_feature(path, rows, day_idx, realized_history=history))
            target = observed_target(row)
            targets.append(math.log1p(target))
            groups.append(path.stem)
            history.append(target)
    return features, targets, groups


def _recursive_predict(model: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    history: list[float] = []
    for day_idx in range(len(rows)):
        feature = row_feature(path, rows, day_idx, realized_history=history)
        prediction = max(1.0, math.expm1(float(model.predict([feature])[0])))
        history.append(prediction)
    return history


def _candidate_factories(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    return {
        "ridge": lambda: Ridge(alpha=2.0),
        "hgb": lambda: HistGradientBoostingRegressor(
            random_state=seed, max_iter=260, learning_rate=0.035,
            max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=0.1,
        ),
        "extra": lambda: ExtraTreesRegressor(
            random_state=seed, n_estimators=400, min_samples_leaf=3,
            max_features=0.85, n_jobs=-1,
        ),
    }


def _pipeline(estimator: Any) -> Any:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import Pipeline

    return Pipeline([("vec", DictVectorizer(sparse=False)), ("model", estimator)])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_bundle(
    train_dir: str | Path,
    output_dir: str | Path,
    *,
    excluded_event_ids: Sequence[str] = DEFAULT_EXCLUDED_EVENT_IDS,
    seed: int = 4925,
) -> Path:
    import joblib
    import numpy as np
    from sklearn.model_selection import GroupKFold

    all_paths = macro_paths(train_dir)
    excluded = set(excluded_event_ids)
    paths = [path for path in all_paths if path.stem not in excluded]
    excluded_present = [path for path in all_paths if path.stem in excluded]
    if not paths or len(paths) < 5:
        raise ValueError("insufficient train events after duplicate exclusion")
    X, y, groups = build_training_rows(paths)
    y_array = np.asarray(y, dtype=np.float64)
    groups_array = np.asarray(groups)
    factories = _candidate_factories(seed)
    cv_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=min(5, len(paths)))
    for candidate, factory in factories.items():
        for fold, (train_index, validation_index) in enumerate(splitter.split(X, y_array, groups_array), start=1):
            model = _pipeline(factory())
            model.fit([X[index] for index in train_index], y_array[train_index])
            validation_groups = sorted(set(groups_array[validation_index]))
            errors: list[float] = []
            for group in validation_groups:
                path = next(path for path in paths if path.stem == group)
                rows = load_jsonl(path)
                predictions = _recursive_predict(model, path, rows)
                truths = [observed_target(row) for row in rows]
                errors.extend(abs(math.log1p(pred) - math.log1p(true)) for pred, true in zip(predictions, truths))
            cv_rows.append({
                "candidate": candidate, "fold": fold,
                "validation_events": len(validation_groups),
                "recursive_log_mae": sum(errors) / len(errors),
            })
    scores = {
        candidate: sum(row["recursive_log_mae"] for row in cv_rows if row["candidate"] == candidate)
        / sum(1 for row in cv_rows if row["candidate"] == candidate)
        for candidate in factories
    }
    best = min(scores, key=scores.get)
    model = _pipeline(factories[best]())
    model.fit(X, y_array)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "causal_autoregressive.joblib"
    bundle = {
        "forecaster_type": FORECASTER_TYPE,
        "variant": VARIANT,
        "target_key": TARGET_KEY,
        "model": model,
        "best_estimator": best,
        "selection_metric": "grouped_event_recursive_log_mae",
        "cv_scores": scores,
        "training_event_count": len(paths),
        "training_row_count": len(X),
        "excluded_duplicate_event_ids": list(excluded_event_ids),
        "excluded_duplicate_files_present": [path.name for path in excluded_present],
        "feature_schema": feature_schema(),
        "forbidden_current_future_inputs": list(FORBIDDEN_FEATURE_TOKENS),
        "train_manifest": directory_manifest(paths),
        "test_data_used_for_fit_or_selection": False,
        "runtime_history": "prior_simulated_writeback_only",
    }
    joblib.dump(bundle, model_path)
    _write_csv(output / "grouped_cv.csv", cv_rows)
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata["model_path"] = str(model_path)
    metadata["model_sha256"] = file_sha256(model_path)
    (output / "forecaster_protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return model_path


def load_causal_autoregressive_bundle(model_path: str | Path) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(model_path)
    if bundle.get("variant") == "causal_ar_train_only_groupcv_v1":
        bundle = dict(bundle, variant=VARIANT)
    if bundle.get("forecaster_type") != FORECASTER_TYPE:
        raise ValueError("not a causal autoregressive forecaster")
    if bundle.get("variant") != VARIANT:
        raise ValueError("causal autoregressive variant drift")
    if bundle.get("test_data_used_for_fit_or_selection") is not False:
        raise ValueError("test data use is forbidden")
    if bundle.get("feature_schema") != feature_schema():
        raise ValueError("causal feature schema drift")
    return bundle


def predict_causal_autoregressive_for_day(
    rows_for_features: Sequence[Mapping[str, Any]],
    day_idx: int,
    bundle: Mapping[str, Any],
    *,
    event_file: str = "event",
) -> dict[str, float | str]:
    history = [observed_target(row) for row in rows_for_features[:day_idx]]
    feature = row_feature(event_file, rows_for_features, day_idx, realized_history=history)
    prediction = max(1.0, math.expm1(float(bundle["model"].predict([feature])[0])))
    return {
        "predicted": prediction,
        "predicted_ratio": 0.0,
        "natural_strength": 0.0,
        "prediction_contract": "train_only_recursive_prior_writeback",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=4925)
    parser.add_argument(
        "--exclude-event", action="append", dest="excluded",
        default=list(DEFAULT_EXCLUDED_EVENT_IDS),
    )
    args = parser.parse_args()
    model_path = train_bundle(
        args.train_dir, args.output_dir,
        excluded_event_ids=tuple(dict.fromkeys(args.excluded)), seed=args.seed,
    )
    print(json.dumps({"model_path": str(model_path), "sha256": file_sha256(model_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
