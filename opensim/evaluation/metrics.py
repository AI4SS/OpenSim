"""Macro metrics and stepwise semantic metrics over all valid matched labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-dir", required=True)
    parser.add_argument("--semantic-csv", required=True)
    parser.add_argument("--teacher-nll-csv", required=True)
    parser.add_argument("--macro-dir", default="data/test/macro")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cad-source", choices=("native_state", "text_classification"), default="native_state")
    return parser.parse_args()


_CLI_ARGS = parse_args() if __name__ == "__main__" else None

import numpy as np
import pandas as pd

from .deterministic import EPS, STATE_KEYS, read_jsonl, parse_dist, target_distribution
from .semantic import DIMENSIONS as MICRO_DIMS, LABELS

SENTIMENT_TO_STATE = {"positive": "pos", "neutral": "neu", "negative": "neg"}
MICRO_MAPPINGS = {
    "rumor": {"spread": 0, "counter": 1},
    "sentiment_state": {"angry": 0, "fear": 1, "sad": 2, "calm": 3, "neutral": 3, "surprise": 4, "happy": 5},
    "sentiment_tendency": {"negative": 0, "neutral": 1, "positive": 2},
    "behavior_type": {"share": 0, "comment": 1},
    "stance": {"oppose": 0, "neutral": 1, "support": 2},
    "belief_degree": {"doubt": 0, "believe": 1},
    "subjectivity": {"objective": 0, "subjective": 1},
    "intent_classification": {"question": 0, "promotion": 1, "opinion": 2},
}


def normalize(arr: Iterable[float]) -> np.ndarray:
    x = np.array(list(arr), dtype=float)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return x / total

def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())

def dtw_numeric(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])

def dtw_distributions_euclidean(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dp = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(dp[i - 1, j], steps[i - 1, j]), (dp[i, j - 1], steps[i, j - 1]), (dp[i - 1, j - 1], steps[i - 1, j - 1])]
            best_cost, best_steps = min(options, key=lambda x: x[0])
            dp[i, j] = float(np.linalg.norm(a[i - 1] - b[j - 1])) + best_cost
            steps[i, j] = best_steps + 1
    return float(dp[n, m] / max(steps[n, m], 1.0))

def kl_div_base2(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p / q) / np.log(2.0))))

def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def micro_f1(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))

def natural_t_key(value: Any) -> int:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0

def attitude_label(dist: np.ndarray) -> str:
    idx = int(np.argmax(dist))
    return STATE_KEYS[idx]

def attitude_for_wass(dist_pos_neu_neg: np.ndarray) -> np.ndarray:
    # Wasserstein assumes ordered sentiment axis: neg -> neu -> pos.
    return np.array([dist_pos_neu_neg[2], dist_pos_neu_neg[1], dist_pos_neu_neg[0]], dtype=float)

def macro_rows_for_event(event: str, macro_dir: Path) -> list[dict[str, Any]]:
    path = macro_dir / f"{event}.jsonl"
    rows = []
    for idx, row in enumerate(read_jsonl(path), start=1):
        rows.append({
            "time_step": f"t{idx}",
            "gt_cumulative_dist": target_distribution(row, "cumulative"),
        })
    return rows

def compute_cad_metrics(event: str, simulation_result: Path, macro_dir: Path) -> dict[str, float]:
    macro_rows = macro_rows_for_event(event, macro_dir)
    sim = pd.read_csv(simulation_result)
    if "source_content_type" in sim.columns:
        sim = sim[sim["source_content_type"].astype(str).str.lower() != "post"].copy()
    cumulative: list[np.ndarray] = []
    gen_cums: list[np.ndarray] = []
    gt_cums: list[np.ndarray] = []
    gen_labels: list[str] = []
    gt_labels: list[str] = []
    last_cum: np.ndarray | None = None
    for row in macro_rows:
        t = row["time_step"]
        day = sim[sim.get("time_step", "") == t]
        day_dists: list[np.ndarray] = []
        for col in ["individual_mu_dist", "mu_dist"]:
            if col in day.columns:
                day_dists = [d for d in (parse_dist(v) for v in day[col].tolist()) if d is not None]
                if day_dists:
                    break
        if day_dists:
            cumulative.extend(day_dists)
            last_cum = normalize(np.mean(np.vstack(cumulative), axis=0))
        gt_cum = row["gt_cumulative_dist"]
        if last_cum is None or not np.isfinite(gt_cum).all():
            continue
        gen_cums.append(last_cum)
        gt_cums.append(gt_cum)
        gen_labels.append(attitude_label(last_cum))
        gt_labels.append(attitude_label(gt_cum))
    if not gen_cums:
        raise ValueError(f"Native CAD has no evaluable support for {event}")
    cad_kl = float(np.mean([kl_div(g, t) for g, t in zip(gen_cums, gt_cums)]))
    cad_w = float(np.mean([wasserstein_1d(attitude_for_wass(g), attitude_for_wass(t)) for g, t in zip(gen_cums, gt_cums)])) if gen_cums else 0.0
    cad_f1 = macro_f1(gt_labels, gen_labels, STATE_KEYS)
    return {"CAD KL": cad_kl, "CAD W": cad_w, "CAD F1": cad_f1, "CAD Support": len(gen_cums)}

def state_dist_from_sentiment_labels(labels: list[str]) -> np.ndarray:
    counts = {key: 0 for key in STATE_KEYS}
    for label in labels:
        state = SENTIMENT_TO_STATE.get(str(label))
        if state:
            counts[state] += 1
    return normalize([counts[key] for key in STATE_KEYS])

def compute_text_cad_metrics(
    event: str,
    semantic: pd.DataFrame,
    macro_dir: Path,
) -> dict[str, float]:
    event_df = semantic[semantic["event"].astype(str) == event].copy()
    macro_rows = macro_rows_for_event(event, macro_dir)
    cumulative_labels: list[str] = []
    gen_cums: list[np.ndarray] = []
    gt_cums: list[np.ndarray] = []
    gen_labels: list[str] = []
    gt_labels: list[str] = []
    last_cum = None
    for row in macro_rows:
        time_step = row["time_step"]
        day = event_df[event_df["time_step"].astype(str) == time_step]
        labels = [
            str(value)
            for value in day.get(
                "gen_sentiment_tendency",
                pd.Series(dtype=str),
            ).tolist()
            if str(value) in SENTIMENT_TO_STATE
        ]
        if labels:
            cumulative_labels.extend(labels)
            last_cum = state_dist_from_sentiment_labels(cumulative_labels)
        gt_cum = row["gt_cumulative_dist"]
        if last_cum is None or not np.isfinite(gt_cum).all():
            continue
        gen_cums.append(last_cum)
        gt_cums.append(gt_cum)
        gen_labels.append(attitude_label(last_cum))
        gt_labels.append(attitude_label(gt_cum))
    if not gen_cums:
        raise ValueError(f"Text CAD has no evaluable support for {event}")
    cad_kl = (
        float(np.mean([kl_div(g, t) for g, t in zip(gen_cums, gt_cums)]))
        if gen_cums
        else 0.0
    )
    cad_w = (
        float(
            np.mean(
                [
                    wasserstein_1d(
                        attitude_for_wass(g),
                        attitude_for_wass(t),
                    )
                    for g, t in zip(gen_cums, gt_cums)
                ]
            )
        )
        if gen_cums
        else 0.0
    )
    cad_f1 = macro_f1(gt_labels, gen_labels, STATE_KEYS)
    return {"CAD KL": cad_kl, "CAD W": cad_w, "CAD F1": cad_f1, "CAD Support": len(gen_cums)}

def compute_stepwise_micro_for_dim(
    df: pd.DataFrame,
    dim: str,
) -> dict[str, float]:
    mapping = MICRO_MAPPINGS[dim]
    real_col = f"real_{dim}"
    gen_col = f"gen_{dim}"
    sub = df[["time_step", real_col, gen_col]].copy()
    sub["y_real"] = sub[real_col].astype(str).map(mapping)
    sub["y_gen"] = sub[gen_col].astype(str).map(mapping)
    sub = sub.dropna(subset=["y_real", "y_gen"])
    if len(sub) == 0:
        return {"KL Div": np.nan, "Wass Dist": np.nan, "DTW": np.nan,
                "Macro F1": np.nan, "Micro F1": np.nan, "Support": 0}
    y_true = sub["y_real"].astype(int).tolist()
    y_pred = sub["y_gen"].astype(int).tolist()
    num_classes = max(mapping.values()) + 1
    label_ids = list(range(num_classes))

    real_seq: list[np.ndarray] = []
    gen_seq: list[np.ndarray] = []
    kl_values: list[float] = []
    wass_values: list[float] = []
    for t in sorted(sub["time_step"].dropna().unique(), key=natural_t_key):
        day = sub[sub["time_step"] == t]
        real = day["y_real"].astype(int).to_numpy()
        gen = day["y_gen"].astype(int).to_numpy()
        if len(real) == 0 or len(gen) == 0:
            continue
        p_real = (np.bincount(real, minlength=num_classes).astype(float) + EPS) / (len(real) + EPS * num_classes)
        p_gen = (np.bincount(gen, minlength=num_classes).astype(float) + EPS) / (len(gen) + EPS * num_classes)
        kl_values.append(kl_div_base2(p_real, p_gen))
        wass_values.append(wasserstein_1d(p_real, p_gen))
        real_seq.append(p_real)
        gen_seq.append(p_gen)

    return {
        "KL Div": float(np.mean(kl_values)) if kl_values else 0.0,
        "Wass Dist": float(np.mean(wass_values)) if wass_values else 0.0,
        "DTW": dtw_distributions_euclidean(real_seq, gen_seq) if len(real_seq) > 1 else 0.0,
        "Macro F1": macro_f1([str(x) for x in y_true], [str(x) for x in y_pred], [str(x) for x in label_ids]),
        "Micro F1": micro_f1([str(x) for x in y_true], [str(x) for x in y_pred]),
        "Support": len(sub),
    }

def compute_micro_for_df(df: pd.DataFrame, event: str, nll_loss: float) -> pd.DataFrame:
    rows = []
    for dim in MICRO_DIMS:
        row = compute_stepwise_micro_for_dim(df, dim)
        row.update({"Event": event, "Dimension": dim, "NLL Loss": nll_loss})
        rows.append(row)
    final = {"Event": event, "Dimension": "final_result",
             "Support": int(np.mean([row["Support"] for row in rows]))}
    for column in ("KL Div", "Wass Dist", "DTW", "Macro F1", "Micro F1", "NLL Loss"):
        final[column] = float(np.mean([row[column] for row in rows]))
    return pd.DataFrame(rows + [final])


def micro_tables(semantic: pd.DataFrame, teacher: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Teacher NLL uses all rows, independently of semantic label availability.
    event_nll = teacher.groupby("event")["content_nll"].mean()
    events = [compute_micro_for_df(frame, str(event), float(event_nll.loc[event]))
              for event, frame in semantic.groupby("event", sort=True)]
    overall = compute_micro_for_df(semantic, "final_result", float(teacher["content_nll"].mean()))
    event_table = pd.concat(events, ignore_index=True) if events else overall.iloc[:0].copy()
    return event_table, overall


def validate_inputs(pairwise: pd.DataFrame, semantic: pd.DataFrame, teacher: pd.DataFrame) -> None:
    from .likelihood import row_uid, validate_scored_rows
    pairs = pairwise.fillna("").to_dict("records")
    expected = [row_uid(row) for row in pairs]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Metric inputs require nonempty unique pairwise identities.")
    for name, frame in (("semantic", semantic), ("teacher NLL", teacher)):
        ids = frame["row_uid"].fillna("").astype(str).tolist()
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError(f"{name} coverage does not match pairwise rows.")
        actual_identity = frame.set_index("row_uid")[["event", "time_step"]].astype(str)
        for row, uid in zip(pairs, expected):
            if tuple(actual_identity.loc[uid]) != (str(row["event"]), str(row["time_step"])):
                raise ValueError(f"{name} event/time identity mismatch: {uid}")
    for dim in MICRO_DIMS:
        for side in ("real", "gen"):
            if not semantic[f"{side}_{dim}"].astype(str).isin(LABELS[dim]).all():
                raise ValueError(f"Missing or invalid semantic label: {side}/{dim}")
    validate_scored_rows(pairs, {str(row["row_uid"]): row for row in teacher.fillna("").to_dict("records")})


def write_outputs(args: argparse.Namespace) -> None:
    det_dir = Path(args.det_dir)
    summary = pd.read_csv(det_dir / "events.csv")
    daily = pd.read_csv(det_dir / "daily.csv")
    pairwise = pd.read_csv(det_dir / "pairwise.csv")
    semantic = pd.read_csv(args.semantic_csv)
    teacher = pd.read_csv(args.teacher_nll_csv)
    validate_inputs(pairwise, semantic, teacher)
    macro_rows = []
    for _, event_row in summary.iterrows():
        event = str(event_row["event"])
        frame = daily[daily["event"].astype(str) == event].sort_values("time_step_index")
        sim_counts = frame["simulated_count"].astype(float).tolist()
        gt_counts = frame["gt_count"].astype(float).tolist()
        if args.cad_source == "native_state":
            cad = compute_cad_metrics(event, Path(str(event_row["simulation_result"])), Path(args.macro_dir))
        else:
            cad = compute_text_cad_metrics(event, semantic, Path(args.macro_dir))
        macro_rows.append({
            "Event": event,
            "DIV KL": kl_div(np.array(sim_counts), np.array(gt_counts)),
            "DIV W": wasserstein_1d(np.array(sim_counts), np.array(gt_counts)),
            "DIV DTW": dtw_numeric(sim_counts, gt_counts), **cad,
        })
    macro = pd.DataFrame(macro_rows)
    macro_final = {"Event": "final_result"}
    macro_final.update(macro.drop(columns="Event").mean().to_dict())
    micro_events, micro = micro_tables(semantic, teacher)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([macro, pd.DataFrame([macro_final])], ignore_index=True).to_csv(out / "macro.csv", index=False)
    micro_events.to_csv(out / "micro_by_event.csv", index=False, na_rep="NA")
    micro.to_csv(out / "micro.csv", index=False, na_rep="NA")
    protocol = {
        "micro_style": "stepwise",
        "micro_kl_units": "bits",
        "macro_kl_units": "nats",
        "epsilon": EPS,
        "kl_direction": {"micro": "observed || generated", "macro": "generated || observed"},
        "micro_aggregation": "unweighted mean over relative time_step distributions; overall pools events",
        "dtw_aggregation": "Euclidean distribution cost divided by path length; relative time steps pooled across events",
        "f1_aggregation": "all valid row-level matched label pairs",
        "nll_source": "per-row teacher-forced content-token NLL, including end marker",
        "nll_aggregation": "mean over all pairwise teacher rows, independent of semantic label availability",
        "cad_source": args.cad_source,
        "state_target_validity": "exclude zero-mass distributions and explicit target_mask=0; report CAD/deterministic valid support",
        "zero_semantic_support": "NA metrics with Support=0; teacher NLL remains available",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _CLI_ARGS or parse_args()
    write_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

