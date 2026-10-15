from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

EPS = 1e-9
STATE_KEYS = ["pos", "neu", "neg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a closed-loop OpenSim run.")
    parser.add_argument("--simulation-result", action="append", default=[])
    parser.add_argument("--saved-paths-json")
    parser.add_argument("--macro-dir", default="data/test/macro")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


_CLI_ARGS = parse_args() if __name__ == "__main__" else None

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_dist(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=float)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        try:
            arr = np.array(ast.literal_eval(text), dtype=float)
        except Exception:
            return None
    if arr.shape != (3,) or not np.isfinite(arr).all() or (arr < 0).any():
        return None
    total = float(arr.sum())
    if total <= 0:
        return None
    return arr / total


def dist_from_dict(value: Any) -> np.ndarray:
    if not isinstance(value, dict):
        return np.full(3, np.nan, dtype=float)
    arr = np.array([float(value.get(k, 0.0) or 0.0) for k in STATE_KEYS], dtype=float)
    total = float(arr.sum())
    if total <= 0 or not np.isfinite(arr).all() or (arr < 0).any():
        return np.full(3, np.nan, dtype=float)
    return arr / total


def target_distribution(row: dict[str, Any], kind: str) -> np.ndarray:
    diagnostics = row.get("diagnostics") or {}
    mask = diagnostics.get(f"{kind}_target_mask", row.get(f"{kind}_target_mask", 1))
    if float(mask) == 0:
        return np.full(3, np.nan, dtype=float)
    if float(mask) != 1:
        raise ValueError(f"Invalid {kind} target mask: {mask}")
    key = f"{kind}_state_distribution"
    return dist_from_dict(row.get(key) or diagnostics.get(key))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum())


def dtw_distance(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def event_from_df(df: pd.DataFrame, csv_path: Path) -> str:
    if "file" in df.columns and len(df) > 0:
        try:
            return Path(str(df["file"].dropna().iloc[0])).stem
        except Exception:
            pass
    if "event" in df.columns and len(df) and df["event"].notna().any():
        return str(df["event"].dropna().iloc[0])
    name = csv_path.stem
    for suffix in ["_mf_start_", "_evaluation_results_"]:
        if suffix in name:
            return name.split(suffix)[0]
    return name


def load_simulation_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.simulation_result]
    if args.saved_paths_json:
        with Path(args.saved_paths_json).open("r", encoding="utf-8") as f:
            for item in json.load(f):
                out = item.get("output_file")
                if out:
                    paths.append(Path(out))
    # preserve order, remove duplicates
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def macro_daily_rows(event: str, macro_dir: Path) -> list[dict[str, Any]]:
    path = macro_dir / f"{event}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"macro not found for {event}: {path}")
    rows = []
    for idx, row in enumerate(read_jsonl(path), start=1):
        diag = row.get("diagnostics") or {}
        sample = int(diag.get("sample_num_interactors", 0) or 0)
        comments = int(diag.get("num_comments", 0) or 0)
        shares = int(diag.get("num_shares", 0) or 0)
        daily_dist = target_distribution(row, "daily")
        cumulative_dist = target_distribution(row, "cumulative")
        rows.append({
            "event": event,
            "time_step": f"t{idx}",
            "time_step_index": idx,
            "gt_count": sample,
            "gt_comment_count": comments,
            "gt_share_count": shares,
            "gt_share_ratio": shares / sample if sample else 0.0,
            "gt_daily_dist": daily_dist,
            "gt_cumulative_dist": cumulative_dist,
        })
    return rows


def evaluate_one(
    csv_path: Path,
    macro_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    event = event_from_df(df, csv_path)
    macro_rows = macro_daily_rows(event, macro_dir)
    macro_by_t = {row["time_step"]: row for row in macro_rows}

    if "source_content_type" not in df.columns:
        df["source_content_type"] = "comment"
    if "generated_action" not in df.columns:
        df["generated_action"] = df.get("generated_comment", pd.Series("", index=df.index)).map(lambda x: "share" if str(x).strip() in {"转发微博", "转发"} else "comment")
    if "time_step" not in df.columns:
        df["time_step"] = "t1"
    if "source_reuse_round" not in df.columns:
        df["source_reuse_round"] = 0

    interaction_df = df[df["source_content_type"].astype(str).str.lower() != "post"].copy()
    interaction_df["generated_action"] = interaction_df["generated_action"].fillna("comment").astype(str)

    daily_rows: list[dict[str, Any]] = []
    cumulative_generated: list[np.ndarray] = []

    for macro in macro_rows:
        t = macro["time_step"]
        day = interaction_df[interaction_df["time_step"] == t]
        sim_count = int(len(day))
        gen_comment = int((day["generated_action"] == "comment").sum()) if sim_count else 0
        gen_share = int((day["generated_action"] == "share").sum()) if sim_count else 0
        gen_share_ratio = gen_share / sim_count if sim_count else 0.0

        dist_values = []
        for col in ["individual_mu_dist", "mu_dist"]:
            if col in day.columns:
                dist_values = [d for d in (parse_dist(v) for v in day[col].tolist()) if d is not None]
                if dist_values:
                    break
        state_prediction_available = bool(dist_values)
        if state_prediction_available:
            gen_daily_dist = np.mean(np.vstack(dist_values), axis=0)
            gen_daily_dist = gen_daily_dist / max(gen_daily_dist.sum(), EPS)
            cumulative_generated.extend(dist_values)
        else:
            gen_daily_dist = np.full(3, np.nan, dtype=float)

        if cumulative_generated:
            gen_cum_dist = np.mean(np.vstack(cumulative_generated), axis=0)
            gen_cum_dist = gen_cum_dist / max(gen_cum_dist.sum(), EPS)
        else:
            gen_cum_dist = np.full(3, np.nan, dtype=float)

        gt_count = int(macro["gt_count"])
        count_abs = abs(sim_count - gt_count)
        count_smape = (2 * count_abs / max(sim_count + gt_count, EPS)) if (sim_count + gt_count) else 0.0
        gt_daily = macro["gt_daily_dist"]
        gt_cum = macro["gt_cumulative_dist"]

        daily_rows.append({
            "event": event,
            "time_step": t,
            "time_step_index": macro["time_step_index"],
            "simulated_count": sim_count,
            "gt_count": gt_count,
            "count_abs_error": count_abs,
            "count_smape": count_smape,
            "generated_comment_count": gen_comment,
            "generated_share_count": gen_share,
            "gt_comment_count": macro["gt_comment_count"],
            "gt_share_count": macro["gt_share_count"],
            "generated_share_ratio": gen_share_ratio,
            "gt_share_ratio": macro["gt_share_ratio"],
            "share_ratio_abs_error": abs(gen_share_ratio - macro["gt_share_ratio"]),
            "state_prediction_available": state_prediction_available,
            "daily_state_valid": bool(np.isfinite(gt_daily).all() and np.isfinite(gen_daily_dist).all()),
            "cumulative_state_valid": bool(np.isfinite(gt_cum).all() and np.isfinite(gen_cum_dist).all()),
            "generated_pos": gen_daily_dist[0],
            "generated_neu": gen_daily_dist[1],
            "generated_neg": gen_daily_dist[2],
            "gt_pos": gt_daily[0],
            "gt_neu": gt_daily[1],
            "gt_neg": gt_daily[2],
            "attitude_kl_daily": kl_div(gen_daily_dist, gt_daily),
            "attitude_js_daily": js_div(gen_daily_dist, gt_daily),
            "attitude_l1_daily": l1_dist(gen_daily_dist, gt_daily),
            "attitude_kl_cumulative": kl_div(gen_cum_dist, gt_cum),
            "attitude_js_cumulative": js_div(gen_cum_dist, gt_cum),
            "attitude_l1_cumulative": l1_dist(gen_cum_dist, gt_cum),
        })

    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty or not daily_df["daily_state_valid"].any() or not daily_df["cumulative_state_valid"].any():
        raise ValueError(f"No evaluable daily/cumulative state support for {event}")

    pairwise = interaction_df.copy()
    pairwise = pairwise[pairwise["source_reuse_round"].fillna(0).astype(int) == 0].copy()
    pairwise["event"] = event
    pairwise["action_match"] = pairwise["source_content_type"].astype(str).str.lower() == pairwise["generated_action"].astype(str).str.lower()

    sim_counts = daily_df["simulated_count"].astype(float).tolist()
    gt_counts = daily_df["gt_count"].astype(float).tolist()
    summary = {
        "event": event,
        "days": len(daily_df),
        "simulated_total": int(daily_df["simulated_count"].sum()),
        "gt_total": int(daily_df["gt_count"].sum()),
        "count_mae": float(daily_df["count_abs_error"].mean()) if len(daily_df) else 0.0,
        "count_rmse": float(np.sqrt(np.mean((daily_df["simulated_count"] - daily_df["gt_count"]) ** 2))) if len(daily_df) else 0.0,
        "count_smape": float(daily_df["count_smape"].mean()) if len(daily_df) else 0.0,
        "count_dtw": dtw_distance(sim_counts, gt_counts) if sim_counts and gt_counts else 0.0,
        "share_ratio_mae": float(daily_df["share_ratio_abs_error"].mean()) if len(daily_df) else 0.0,
        "attitude_state_coverage": float(daily_df["state_prediction_available"].mean()) if len(daily_df) else 0.0,
        "attitude_daily_support": int(daily_df["daily_state_valid"].sum()),
        "attitude_cumulative_support": int(daily_df["cumulative_state_valid"].sum()),
        "attitude_daily_kl_mean": float(daily_df["attitude_kl_daily"].mean()) if len(daily_df) else 0.0,
        "attitude_daily_js_mean": float(daily_df["attitude_js_daily"].mean()) if len(daily_df) else 0.0,
        "attitude_cumulative_kl_mean": float(daily_df["attitude_kl_cumulative"].mean()) if len(daily_df) else 0.0,
        "attitude_cumulative_js_mean": float(daily_df["attitude_js_cumulative"].mean()) if len(daily_df) else 0.0,
        "pairwise_action_acc": float(pairwise["action_match"].mean()) if len(pairwise) else 0.0,
        "pairwise_coverage": float(len(pairwise) / max(int(daily_df["gt_count"].sum()), 1)),
        "copy_rate": float((pairwise["generated_comment"].astype(str) == pairwise["real_comment"].astype(str)).mean()) if len(pairwise) else 0.0,
        "empty_generation_rate": float(pairwise["generated_comment"].fillna("").astype(str).str.strip().eq("").mean()) if len(pairwise) else 0.0,
        "failure_generation_rate": float(pairwise["generated_comment"].fillna("").astype(str).str.contains("调用失败|失败", regex=True).mean()) if len(pairwise) else 0.0,
        "simulation_result": str(csv_path),
    }
    return daily_df, pd.DataFrame([summary]), pairwise


def validate_evaluation_frames(
    daily_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    expected_results: int,
) -> None:
    if expected_results <= 0:
        raise ValueError("Deterministic evaluation has no input results.")
    if summary_df.empty or pairwise_df.empty or daily_df.empty:
        raise ValueError("Deterministic evaluation produced an empty artifact.")
    summary_events = set(summary_df["event"].dropna().astype(str))
    daily_events = set(daily_df["event"].dropna().astype(str))
    pairwise_events = set(pairwise_df["event"].dropna().astype(str))
    if (
        len(summary_df) != expected_results
        or len(summary_events) != expected_results
        or daily_events != summary_events
        or pairwise_events != summary_events
    ):
        raise ValueError(
            "Deterministic evaluation event coverage mismatch: "
            f"inputs={expected_results} summaries={len(summary_df)} "
            f"summary_events={len(summary_events)} "
            f"daily_events={len(daily_events)} "
            f"pairwise_events={len(pairwise_events)}"
        )
    required_runtime_fields = (
        "source_sampling_mode",
        "source_sampling_seed",
        "post_mode",
        "policy_format",
        "policy_prompt",
    )
    incomplete = {}
    for field in required_runtime_fields:
        if field not in pairwise_df.columns:
            incomplete[field] = len(pairwise_df)
            continue
        missing = int(
            pairwise_df[field]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        if missing:
            incomplete[field] = missing
    if incomplete:
        raise ValueError(
            f"Deterministic runtime audit fields are incomplete: {incomplete}"
        )


def main() -> None:
    args = _CLI_ARGS or parse_args()
    sim_paths = load_simulation_paths(args)
    if not sim_paths:
        raise SystemExit("No simulation result provided")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    macro_dir = Path(args.macro_dir)

    all_daily = []
    all_summary = []
    all_pairwise = []

    for path in sim_paths:
        if not path.exists():
            raise FileNotFoundError(f"Simulation CSV missing: {path}")
        print(f"[eval] {path}")
        daily, summary, pairwise = evaluate_one(
            path,
            macro_dir,
        )
        all_daily.append(daily)
        all_summary.append(summary)
        all_pairwise.append(pairwise)

    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    pairwise_df = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    validate_evaluation_frames(
        daily_df,
        summary_df,
        pairwise_df,
        expected_results=len(sim_paths),
    )

    daily_df.to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(out_dir / "pairwise.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote evaluation to {out_dir}")
    if not summary_df.empty:
        print(summary_df[["event", "simulated_total", "gt_total", "count_mae", "share_ratio_mae", "pairwise_action_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()

