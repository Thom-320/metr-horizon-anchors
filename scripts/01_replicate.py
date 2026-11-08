"""Step 1 (K1 replication gate): reproduce METR's published TH1.1 p50/p80 with METR's own code."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import mhv_common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region  # noqa: E402
from horizon.wrangle.bootstrap import compute_bootstrap_regressions  # noqa: E402
from horizon.wrangle.logistic import run_logistic_regressions  # noqa: E402

REFERENCE = {  # gate models: one recent frontier + older ones
    "Claude Opus 4.6 (Inspect)": "claude_opus_4_6_inspect",
    "GPT-4 0314": "gpt_4",
    "Claude 3.7 Sonnet (Inspect)": "claude_3_7_sonnet_inspect",
    "o3 (Inspect)": "o3_inspect",
}


def main() -> None:
    runs = C.load_runs()
    params = {"runs_file": C.RUNS_TH11, "weighting": C.WEIGHTING, "categories": "ftr",
              "regularization": C.REGULARIZATION, "exclude": [], "success_percents": C.SUCCESS_PERCENTS,
              "confidence_level": 0.95}
    # METR bootstrap stage (same call as dvc stage wrangle_bootstrap_logistic@headline)
    data = runs.rename(columns={"alias": "agent"})
    boot = compute_bootstrap_regressions(data=data, categories=["task_family", "task_id", "run_id"],
                                         n_bootstrap=C.N_BOOTSTRAP, regularization=C.REGULARIZATION,
                                         weights_col=C.WEIGHTING, success_percents=C.SUCCESS_PERCENTS,
                                         score_col="score_binarized")
    boot_path = C.RESULTS / "metr_headline_bootstrap.csv.gz"
    boot.to_csv(boot_path, index=False)
    bootfile = C.RESULTS / "_tmp_boot.csv"
    boot.to_csv(bootfile, index=False)
    fits = run_logistic_regressions(runs, C.RELEASE_DATES, params, bootfile)
    bootfile.unlink()
    fits.to_csv(C.RESULTS / "metr_headline_fits_reproduced.csv", index=False)

    pub = yaml.safe_load(C.PUBLISHED_TH11.read_text())
    model_of_alias = runs.drop_duplicates("alias").set_index("alias")["model"].to_dict()
    rows = []
    for _, r in fits.iterrows():
        agent = r["agent"]
        if agent == "human":
            continue
        key = model_of_alias[agent]
        key = {"flamingo_2": "gpt_5_3_codex"}.get(key, key)  # GPT-5.3-Codex runs carry an internal model id
        pm = pub["results"].get(key, {}).get("metrics")
        if pm is None:
            continue
        p50p, p80p = pm["p50_horizon_length"], pm["p80_horizon_length"]
        rows.append({
            "agent": agent, "published_key": key, "release_date": r["release_date"],
            "reference_model": agent in REFERENCE,
            "p50_published": p50p["estimate"], "p50_reproduced": r["p50"],
            "p50_rel_diff": r["p50"] / p50p["estimate"] - 1,
            "p50_ci_published": f"[{p50p['ci_low']:.2f}, {p50p['ci_high']:.2f}]",
            "p50_ci_reproduced": f"[{r['p50q0.025']:.2f}, {r['p50q0.975']:.2f}]",
            "p80_published": p80p["estimate"], "p80_reproduced": r["p80"],
            "p80_rel_diff": r["p80"] / p80p["estimate"] - 1,
            "within_10pct": abs(r["p50"] / p50p["estimate"] - 1) <= 0.10,
        })
    gate = pd.DataFrame(rows).sort_values("p50_published")
    gate.to_csv(C.RESULTS / "replication_gate.csv", index=False)

    # Doubling time, from_2023_on, as in METR's generate_benchmark_results._get_trend_stats
    rd = C.load_release_dates()
    summ = fits[fits.agent != "human"].copy()
    summ["release_date"] = summ["agent"].map(rd["date"])
    stats, *_ = compute_bootstrap_confidence_region(
        agent_summaries=summ, bootstrap_results=boot.drop(columns=[c for c in boot.columns if c.startswith("human_")]),
        release_dates=rd, after_date=C.TREND_AFTER, sota_before_date=C.TREND_BEFORE,
        trendline_end_date=C.TREND_BEFORE, confidence_level=0.95)
    dpub = pub["doubling_time_in_days"]["from_2023_on"]
    dt = {"published_point_days": dpub["point_estimate"], "published_ci": [dpub["ci_low"], dpub["ci_high"]],
          "reproduced_point_days": round(stats.point_estimate, 3),
          "reproduced_ci": [round(stats.ci_lower, 3), round(stats.ci_upper, 3)],
          "rel_diff_point": stats.point_estimate / dpub["point_estimate"] - 1,
          "sota_agents": stats.sota_agents, "n_bootstrap_valid": stats.n_samples}
    ref = gate[gate.reference_model]
    k1 = {"reference_models": ref.agent.tolist(),
          "max_abs_rel_diff_p50_reference": float(ref.p50_rel_diff.abs().max()),
          "max_abs_rel_diff_p50_all_models": float(gate.p50_rel_diff.abs().max()),
          "n_models_compared": int(len(gate)),
          "K1_pass": bool(ref.within_10pct.all() and len(ref) >= 3),
          "doubling_from_2023": dt}
    (C.RESULTS / "replication_summary.json").write_text(json.dumps(k1, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(gate[["agent", "p50_published", "p50_reproduced", "p50_rel_diff", "p50_ci_published",
                "p50_ci_reproduced", "p80_rel_diff"]].to_string(index=False))
    print(json.dumps(k1, indent=2, default=str))


if __name__ == "__main__":
    main()
