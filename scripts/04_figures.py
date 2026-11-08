"""Figures + compact memo tables from the results CSVs."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import mhv_common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
SHOW = ["GPT-4 0314", "Claude 3.7 Sonnet (Inspect)", "o3 (Inspect)", "GPT-5 (Inspect)",
        "Claude Opus 4.5 (Inspect)", "GPT-5.2", "Claude Opus 4.6 (Inspect)"]


def fig1() -> None:
    g = pd.read_csv(C.RESULTS / "replication_gate.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.loglog(g.p50_published / 60, g.p50_reproduced / 60, "o", ms=5,
              mfc="none", label="all 20 public TH1.1 models")
    r = g[g.reference_model]
    ax.loglog(r.p50_published / 60, r.p50_reproduced / 60, "o", ms=7, color="C3", label="gate reference models")
    lim = [g.p50_published.min() / 60 / 2, g.p50_published.max() / 60 * 2]
    ax.plot(lim, lim, "k-", lw=.7)
    ax.fill_between(lim, [l * .9 for l in lim], [l * 1.1 for l in lim], color="grey", alpha=.2, label="±10 % (K1)")
    for _, x in r.iterrows():
        ax.annotate(x.agent.replace(" (Inspect)", ""), (x.p50_published / 60, x.p50_reproduced / 60),
                    fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("published p50 (h), metr.org benchmark_results_1_1.yaml")
    ax.set_ylabel("reproduced p50 (h), METR code @52cb829")
    ax.set_title("K1 replication gate: max |rel. diff| = %.1e" % g.p50_rel_diff.abs().max())
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(C.FIGURES / "fig1_replication_gate.png"); plt.close(fig)


def fig2() -> None:
    h = pd.read_csv(C.RESULTS / "horizons_by_variant.csv", parse_dates=["release_date"])
    d = pd.read_csv(C.RESULTS / "doubling_by_variant.csv")
    t = pd.read_csv(C.RESULTS / "anchor_counts_by_length.csv", index_col=0)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    t[["baseline", "estimate"]].plot.bar(stacked=True, ax=a0, color=["C0", "C1"])
    a0.set_title("TH1.1 tasks by human-time source"); a0.set_ylabel("tasks"); a0.set_xlabel("")
    for i, (b, e) in enumerate(zip(t.baseline, t.estimate)):
        a0.text(i, b + e + 1, f"{b}/{e}", ha="center", fontsize=7)
    for v, col, off in (("all", "C0", -3), ("baseline_only", "C1", 3)):
        s = h[h.variant == v].sort_values("release_date")
        x = s.release_date + pd.to_timedelta(off, "D")
        hi = np.minimum(s.p50_hi, 1e5)
        a1.errorbar(x, s.p50 / 60, yerr=[(s.p50 - s.p50_lo) / 60, (hi - s.p50) / 60], fmt="o", ms=4, color=col,
                    capsize=2, lw=.8, label=None)
        r = d[(d.variant == v) & (d.sota_rule == "fixed_sota_all")].iloc[0]
        ss = s[s.agent.isin(r.sota_agents.split(";")) & (s.release_date >= C.TREND_AFTER)]
        xn = matplotlib.dates.date2num(ss.release_date)
        b1, b0 = np.polyfit(xn, np.log(ss.p50), 1)
        xx = np.linspace(xn.min(), xn.max(), 50)
        a1.plot(matplotlib.dates.num2date(xx), np.exp(b0 + b1 * xx) / 60, color=col, lw=1,
                label=f"{v}: doubling {r.doubling_days:.0f} d [{r.ci_lo:.0f}, {r.ci_hi:.0f}]")
    a1.axhline(C.LONG_MIN / 60, color="grey", ls=":", lw=.8)
    a1.axhline(1111.887 / 60, color="C1", ls=":", lw=.8)
    a1.text(pd.Timestamp("2023-02-01"), 1111.887 / 60 * 1.1, "longest baselined task (18.5 h)", fontsize=7, color="C1")
    a1.set_yscale("log"); a1.set_ylabel("p50 time horizon (h), 95 % task-bootstrap CI (capped)")
    a1.set_title("Anchor split: all tasks vs baselined-only tasks (SOTA set fixed, from 2023)")
    a1.set_ylim(0.01, 2000); a1.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(C.FIGURES / "fig2_anchor_split.png"); plt.close(fig)


def fig3() -> None:
    h = pd.read_csv(C.RESULTS / "horizons_by_variant.csv")
    d = pd.read_csv(C.RESULTS / "doubling_by_variant.csv")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    ks = [1.0] + C.K_VALUES
    for fam, mk in (("estall", "o-"), ("estlong", "s--")):
        vs = ["all"] + [f"{fam}_k{k}" for k in C.K_VALUES]
        for agent, col in ((C.FRONTIER, "C3"), ("GPT-5 (Inspect)", "C2"), ("Claude 3.7 Sonnet (Inspect)", "C0")):
            s = h[h.agent == agent].set_index("variant").loc[vs]
            order = np.argsort(ks)
            kk = np.array(ks)[order]; r = s.p50_ratio_vs_all.to_numpy()[order]
            lo = s.p50_ratio_lo.to_numpy()[order]; hi = s.p50_ratio_hi.to_numpy()[order]
            a0.plot(kk, r, mk, color=col, ms=4, label=f"{agent.replace(' (Inspect)', '')} ({fam})")
            if fam == "estall":
                a0.fill_between(kk, lo, hi, color=col, alpha=.12)
        s = d[d.sota_rule == "fixed_sota_all"].set_index("variant").loc[vs]
        order = np.argsort(ks); kk = np.array(ks)[order]
        a1.plot(kk, s.ratio_vs_all.to_numpy()[order], mk, color="k", ms=4, label=f"{fam}: estimate tasks ×k")
        a1.fill_between(kk, s.ratio_lo.to_numpy()[order], s.ratio_hi.to_numpy()[order], color="grey", alpha=.15)
    for a, thr, lab in ((a0, .15, "±15 % (K2, p50)"), (a1, .10, "±10 % (K2, doubling)")):
        a.axhspan(1 - thr, 1 + thr, color="gold", alpha=.25, label=lab)
        a.axvspan(*C.K2_RANGE, color="grey", alpha=.07)
        a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels([str(k) for k in ks])
        a.set_xlabel("k (multiplier on estimated human times); grey = K2 range [0.5, 2]")
    a0.set_ylabel("p50 / p50(k=1)"); a0.set_title("p50 change, 95 % paired task-bootstrap band")
    a1.set_ylabel("doubling time / doubling(k=1)"); a1.set_title("Frontier-trend doubling time (SOTA from 2023, fixed set)")
    a0.legend(fontsize=6.5); a1.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIGURES / "fig3_k_sensitivity.png"); plt.close(fig)


def tables() -> None:
    h = pd.read_csv(C.RESULTS / "horizons_by_variant.csv")
    t = h[h.agent.isin(SHOW)].copy()
    t["p50_h"] = (t.p50 / 60).round(2)
    t["p50_ratio [95% CI]"] = t.apply(lambda r: f"{r.p50_ratio_vs_all:.2f} [{r.p50_ratio_lo:.2f}, {r.p50_ratio_hi:.2f}]", axis=1)
    t["p80_h"] = (t.p80 / 60).round(2)
    t["p80_ratio [95% CI]"] = t.apply(lambda r: f"{r.p80_ratio_vs_all:.2f} [{r.p80_ratio_lo:.2f}, {r.p80_ratio_hi:.2f}]", axis=1)
    t[["variant", "agent", "p50_h", "p50_ratio [95% CI]", "p80_h", "p80_ratio [95% CI]"]].to_csv(
        C.RESULTS / "memo_table_horizons.csv", index=False)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); tables()
    print("figures:", sorted(p.name for p in C.FIGURES.glob("*.png")))
