"""Step 2: anchor split (baselined vs estimated human times) and k-sensitivity, with a paired
hierarchical task bootstrap (METR's bootstrap_sample: family -> task -> run, seeds 42+i)."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import mhv_common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from horizon.plot.bootstrap_ci import (  # noqa: E402
    _compute_doubling_time_and_predictions_from_p50s, get_sota_agents)
from horizon.wrangle.bootstrap import bootstrap_sample  # noqa: E402

CATS = ["task_family", "task_id", "run_id"]
TP = pd.date_range("2023-01-01", "2023-01-02", freq="D")  # predictions unused; only the slope matters


def classify_tasks(runs: pd.DataFrame) -> pd.DataFrame:
    t = runs.drop_duplicates("task_id")[["task_id", "task_family", "task_source", "human_minutes", "human_source"]]
    bins = [0, 1, 4, 16, 60, 240, 480, 960, np.inf]
    labels = ["<1m", "1-4m", "4-16m", "16-60m", "1-4h", "4-8h", "8-16h", ">=16h"]
    t = t.assign(length_bucket=pd.cut(t.human_minutes, bins=bins, labels=labels, right=False))
    return t.sort_values("human_minutes")


def one_boot(i: int, data: pd.DataFrame, agents: list[str]) -> dict:
    rng = np.random.default_rng(C.SEED_BASE + i)
    s = bootstrap_sample(data, CATS, rng)
    s = s[s.agent.isin(agents)]
    out = {"sample": i}
    for v in C.VARIANTS:
        for agent, h in C.fit_variant(s, v, ensure_sum1=False).items():
            out[f"{v}|{agent}|p50"] = h["p50"]
            out[f"{v}|{agent}|p80"] = h["p80"]
    return out


def doubling(p50_by_agent: dict[str, float], agents: list[str], dates: dict[str, str]) -> float:
    pts = [(p50_by_agent[a], dates[a]) for a in agents
           if a in p50_by_agent and np.isfinite(p50_by_agent[a]) and p50_by_agent[a] >= 1e-3]
    if len(pts) < 2:
        return np.nan
    dt, _ = _compute_doubling_time_and_predictions_from_p50s(pts, TP)
    return dt


def main() -> None:
    runs = C.load_runs()
    tasks = classify_tasks(runs)
    tasks.to_csv(C.RESULTS / "task_anchor_classification.csv", index=False)
    counts = tasks.groupby(["length_bucket", "human_source"], observed=False).size().unstack(fill_value=0)
    counts.to_csv(C.RESULTS / "anchor_counts_by_length.csv")
    print(counts)

    data = C.add_variant_columns(runs).rename(columns={"alias": "agent"})
    agents = sorted(a for a in data.agent.unique() if a != "human")
    dates = C.load_release_dates()["date"]

    # weight share of estimate-anchored and long tasks, per agent (METR weights)
    sh = []
    for a, d in data[data.agent.isin(agents)].groupby("agent"):
        w = d.w_all / d.w_all.sum()
        sh.append({"agent": a, "weight_share_estimate": float(w[d.human_source == "estimate"].sum()),
                   "weight_share_ge8h": float(w[d.human_minutes >= C.LONG_MIN].sum()),
                   "weight_share_ge8h_estimate": float(w[(d.human_minutes >= C.LONG_MIN) & (d.human_source == "estimate")].sum()),
                   "n_runs": len(d)})
    pd.DataFrame(sh).to_csv(C.RESULTS / "weight_shares.csv", index=False)

    # point estimates (weights sum to 1 per agent for every variant)
    pdata = data[data.agent.isin(agents)]
    point = {v: C.fit_variant(pdata, v, ensure_sum1=True) for v in C.VARIANTS}

    # paired bootstrap: same resample, all variants
    batches = [list(range(s, min(s + 10, C.N_BOOTSTRAP))) for s in range(0, C.N_BOOTSTRAP, 10)]
    res = Parallel(n_jobs=-1, verbose=0)(
        delayed(lambda b: [one_boot(i, data, agents) for i in b])(b) for b in batches)
    boot = \
        pd.DataFrame([r for b in res for r in b]).sort_values("sample").reset_index(drop=True)
    boot.to_csv(C.RESULTS / "bootstrap_variants.csv.gz", index=False)

    # horizons table with paired ratio CIs vs 'all'
    rows = []
    for v in C.VARIANTS:
        for a in agents:
            r = {"variant": v, "agent": a, "release_date": dates[a]}
            for p in ("p50", "p80"):
                b = boot[f"{v}|{a}|{p}"]
                ball = boot[f"all|{a}|{p}"]
                ratio = (b / ball).replace([np.inf, -np.inf], np.nan)
                r.update({p: point[v][a][p], f"{p}_lo": np.nanquantile(b, .025), f"{p}_hi": np.nanquantile(b, .975),
                          f"{p}_ratio_vs_all": point[v][a][p] / point["all"][a][p],
                          f"{p}_ratio_lo": np.nanquantile(ratio, .025), f"{p}_ratio_hi": np.nanquantile(ratio, .975)})
            rows.append(r)
    hz = pd.DataFrame(rows)
    hz.to_csv(C.RESULTS / "horizons_by_variant.csv", index=False)

    # doubling time per variant (from 2023): (1) METR's SOTA rule re-applied to the variant;
    # (2) fixed SOTA set from the 'all' fit. Bootstrap CIs; paired ratio vs 'all'.
    def sota_for(v: str) -> list[str]:
        s = pd.DataFrame({"agent": agents, "p50": [point[v][a]["p50"] for a in agents]})
        return get_sota_agents(s, dates, C.TREND_AFTER, C.TREND_BEFORE)

    sota_all = sota_for("all")
    drows = []
    per_sample = {}
    for v in C.VARIANTS:
        # third rule mimics the note in metr.org benchmark_results_1_1.yaml ("excludes points with central
        # estimate p50 > 16 hrs"); that filter is not in the public code at 52cb829, so it is reconstructed here.
        excl16 = [a for a in sota_all if point[v][a]["p50"] <= 960]
        for rule, sset in (("variant_sota", sota_for(v)), ("fixed_sota_all", sota_all),
                           ("fixed_sota_all_excl_p50_gt16h", excl16)):
            pt = doubling({a: point[v][a]["p50"] for a in agents}, sset, dates)
            samp = np.array([doubling({a: boot.at[i, f"{v}|{a}|p50"] for a in sset}, sset, dates)
                             for i in range(len(boot))])
            per_sample[(v, rule)] = samp
            pos = samp[samp > 0]
            drows.append({"variant": v, "sota_rule": rule, "n_sota": len(sset), "sota_agents": ";".join(sset),
                          "doubling_days": pt, "ci_lo": np.percentile(pos, 2.5), "ci_hi": np.percentile(pos, 97.5),
                          "n_valid": len(pos)})
    dt = pd.DataFrame(drows)
    for idx, r in dt.iterrows():
        base = dt[(dt.variant == "all") & (dt.sota_rule == r.sota_rule)].iloc[0]
        ratio = per_sample[(r.variant, r.sota_rule)] / per_sample[("all", r.sota_rule)]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        dt.loc[idx, "ratio_vs_all"] = r.doubling_days / base.doubling_days
        dt.loc[idx, "ratio_lo"] = np.percentile(ratio, 2.5)
        dt.loc[idx, "ratio_hi"] = np.percentile(ratio, 97.5)
    dt.to_csv(C.RESULTS / "doubling_by_variant.csv", index=False)

    # K2 read-out: anchor error k in [0.5, 2]; thresholds 15 % on frontier p50, 10 % on doubling
    k2 = []
    for v in [x for x in C.VARIANTS if "_k" in x] + ["baseline_only"]:
        f = hz[(hz.variant == v) & (hz.agent == C.FRONTIER)].iloc[0]
        d = dt[(dt.variant == v) & (dt.sota_rule == "fixed_sota_all")].iloc[0]
        d2 = dt[(dt.variant == v) & (dt.sota_rule == "variant_sota")].iloc[0]
        kv = float(v.split("_k")[1]) if "_k" in v else np.nan
        k2.append({"variant": v, "k": kv, "in_K2_range": bool(C.K2_RANGE[0] <= kv <= C.K2_RANGE[1]) if "_k" in v else False,
                   "frontier": C.FRONTIER, "frontier_p50_change": f.p50_ratio_vs_all - 1,
                   "frontier_p50_change_ci": f"[{f.p50_ratio_lo - 1:+.3f}, {f.p50_ratio_hi - 1:+.3f}]",
                   "doubling_change_fixed_sota": d.ratio_vs_all - 1,
                   "doubling_change_fixed_sota_ci": f"[{d.ratio_lo - 1:+.3f}, {d.ratio_hi - 1:+.3f}]",
                   "doubling_change_variant_sota": d2.ratio_vs_all - 1,
                   "p50_moves_ge15pct": abs(f.p50_ratio_vs_all - 1) >= 0.15,
                   "doubling_moves_ge10pct": abs(d.ratio_vs_all - 1) >= 0.10 or abs(d2.ratio_vs_all - 1) >= 0.10})
    k2 = pd.DataFrame(k2)
    rng = k2[k2.in_K2_range]
    summary = {"K2_range": C.K2_RANGE, "max_abs_frontier_p50_change_in_range": float(rng.frontier_p50_change.abs().max()),
               "max_abs_doubling_change_in_range_fixed_sota": float(rng.doubling_change_fixed_sota.abs().max()),
               "max_abs_doubling_change_in_range_variant_sota": float(rng.doubling_change_variant_sota.abs().max()),
               "p50_threshold": 0.15, "doubling_threshold": 0.10}
    summary["p50_criterion_met"] = summary["max_abs_frontier_p50_change_in_range"] >= 0.15
    summary["doubling_criterion_met"] = max(summary["max_abs_doubling_change_in_range_fixed_sota"],
                                            summary["max_abs_doubling_change_in_range_variant_sota"]) >= 0.10
    (C.RESULTS / "k2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    k2.to_csv(C.RESULTS / "k2_readout.csv", index=False)

    pd.set_option("display.width", 250)
    show = hz[hz.agent.isin([C.FRONTIER, "GPT-5.2", "Claude Opus 4.5 (Inspect)", "GPT-5 (Inspect)",
                             "o3 (Inspect)", "Claude 3.7 Sonnet (Inspect)", "GPT-4 0314"])]
    print(show[["variant", "agent", "p50", "p50_lo", "p50_hi", "p50_ratio_vs_all", "p50_ratio_lo", "p50_ratio_hi",
                "p80", "p80_ratio_vs_all"]].round(3).to_string(index=False))
    print(dt.drop(columns="sota_agents").round(3).to_string(index=False))
    print(k2.round(3).to_string(index=False))
    print(pd.DataFrame(sh).round(3).to_string(index=False))
    json.dump({"sota_all": sota_all, "sota_baseline_only": sota_for("baseline_only")},
              open(C.RESULTS / "sota_sets.json", "w"), indent=2)


if __name__ == "__main__":
    main()
