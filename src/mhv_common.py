"""Shared helpers for the METR time-horizon validity pilot.

All model fitting goes through METR's own functions (vendored at a pinned commit):
  horizon.utils.logistic.logistic_regression / get_x_for_quantile
  horizon.wrangle.logistic.run_logistic_regressions
  horizon.wrangle.bootstrap.bootstrap_sample   (family -> task -> run hierarchical resample)
  horizon.compute_task_weights.add_task_weight_columns
  horizon.plot.bootstrap_ci.get_sota_agents / _compute_doubling_time_and_predictions_from_p50s
"""
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "eval-analysis-public"
RUNS_TH11 = VENDOR / "reports" / "time-horizon-1-1" / "data" / "raw" / "runs.jsonl"
RELEASE_DATES = VENDOR / "data" / "external" / "release_dates.yaml"
PUBLISHED_TH11 = ROOT / "published" / "benchmark_results_1_1.yaml"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# horizon.utils.plots imports cairosvg (only used to draw METR's logo watermark).
# libcairo is not on the default macOS dyld path, so stub it: no numeric code uses it.
if "cairosvg" not in sys.modules:
    sys.modules["cairosvg"] = types.ModuleType("cairosvg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from horizon.compute_task_weights import add_task_weight_columns  # noqa: E402
from horizon.utils.logistic import get_x_for_quantile, logistic_regression  # noqa: E402

# METR headline parameters (reports/time-horizon-1-1/fig_params/figs.yaml, wrangle_logistic.headline)
WEIGHTING = "invsqrt_task_weight"
REGULARIZATION = 1e-5
SUCCESS_PERCENTS = [50, 80]
N_BOOTSTRAP = int(__import__("os").environ.get("MHV_NBOOT", "1000"))  # >= 1000 for reported results
SEED_BASE = 42  # METR uses np.random.default_rng(42 + i) for bootstrap i
TREND_AFTER = "2023-01-01"  # 'from_2023_on' in METR's generate_benchmark_results.py
TREND_BEFORE = "2030-01-01"
LONG_MIN = 480  # 8 h
FRONTIER = "Claude Opus 4.6 (Inspect)"
K_VALUES = [0.5, 0.75, 1.5, 2.0, 3.0, 4.0]  # task spec: {0.5, 0.75, 1.5, 2}; 3 and 4 added as out-of-range context
K2_RANGE = (0.5, 2.0)


def load_runs() -> pd.DataFrame:
    return pd.read_json(RUNS_TH11, lines=True, orient="records", convert_dates=False)


def load_release_dates() -> dict:
    return yaml.safe_load(RELEASE_DATES.read_text())


def fit_horizons(minutes: np.ndarray, y: np.ndarray, w: np.ndarray,
                 ensure_sum1: bool) -> dict[str, float]:
    """METR's per-agent fit: weighted logistic on log2(minutes), C = 1/1e-5."""
    out = {f"p{p}": np.nan for p in SUCCESS_PERCENTS}
    if len(np.unique(y)) < 2:
        return out
    x = np.log2(minutes).reshape(-1, 1)
    model = logistic_regression(x, y, sample_weight=w, regularization=REGULARIZATION,
                                ensure_weights_sum_to_1=ensure_sum1)
    for p in SUCCESS_PERCENTS:
        out[f"p{p}"] = float(np.exp2(get_x_for_quantile(model, p / 100)))
    out["slope"] = float(model.coef_[0][0])
    return out


def add_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-variant x (minutes) and weight columns to the run table.

    all            : METR headline (all tasks, METR's invsqrt weights)
    baseline_only  : only tasks with human_source == 'baseline'; weights recomputed with
                     METR's add_task_weight_columns on that subset (as METR does when it excludes tasks)
    estall_k{k}    : every estimate-sourced task's human_minutes multiplied by k
    estlong_k{k}   : only estimate-sourced tasks with human_minutes >= 480 multiplied by k
    """
    df = df.copy()
    est = df["human_source"].eq("estimate")
    longest = est & df["human_minutes"].ge(LONG_MIN)
    df["x_all"] = df["human_minutes"]
    df["w_all"] = df[WEIGHTING]
    base = df[~est].drop(columns=["equal_task_weight", "invsqrt_task_weight"])
    base = add_task_weight_columns(base)
    df["w_baseline_only"] = np.nan
    df.loc[base.index, "w_baseline_only"] = base["invsqrt_task_weight"]
    df["x_baseline_only"] = df["human_minutes"].where(~est)
    for k in K_VALUES:
        df[f"x_estall_k{k}"] = np.where(est, df["human_minutes"] * k, df["human_minutes"])
        df[f"w_estall_k{k}"] = df[WEIGHTING]
        df[f"x_estlong_k{k}"] = np.where(longest, df["human_minutes"] * k, df["human_minutes"])
        df[f"w_estlong_k{k}"] = df[WEIGHTING]
    return df


VARIANTS = ["all", "baseline_only"] + [f"estall_k{k}" for k in K_VALUES] + [f"estlong_k{k}" for k in K_VALUES]


def fit_variant(df: pd.DataFrame, variant: str, ensure_sum1: bool) -> dict[str, dict[str, float]]:
    res = {}
    xcol, wcol = f"x_{variant}", f"w_{variant}"
    for agent, d in df.groupby("agent"):
        d = d[d[xcol].notna()]
        res[agent] = fit_horizons(d[xcol].to_numpy(float), d["score_binarized"].to_numpy(float),
                                  d[wcol].to_numpy(float), ensure_sum1)
    return res
