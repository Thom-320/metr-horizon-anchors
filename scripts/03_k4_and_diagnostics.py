"""K4 check (how disqualified/illegitimate runs are encoded) and a descriptive 'implied k' diagnostic."""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import mhv_common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from horizon.wrangle.bootstrap import bootstrap_sample  # noqa: E402

PAT = re.compile(r"disqualif|cheat|reward.?hack|illegitim|exclude_run|manual_review|flagged", re.I)


def k4() -> dict:
    hits = []
    for f in sorted((C.VENDOR / "src").rglob("*.py")) + sorted((C.VENDOR / "reports").rglob("*.yaml")):
        for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if PAT.search(line):
                hits.append(f"{f.relative_to(C.VENDOR)}:{n}: {line.strip()}")
    runs = C.load_runs()
    runs["sc"] = runs.score_cont.round(6)
    mixed = int((runs.groupby(["task_id", "sc"]).score_binarized.nunique() > 1).sum())
    ct = pd.crosstab(runs.fatal_error_from.fillna("none"), runs.score_binarized)
    long_succ = runs[(runs.human_minutes >= C.LONG_MIN) & (runs.score_binarized == 1) & (runs.alias != "human")]
    return {
        "columns": list(runs.columns.drop("sc")),
        "code_hits_for_disqualification_terms": hits,
        "n_task_scorecont_pairs_with_mixed_binarized": mixed,
        "bin1_with_score_cont_below_0.5": int(((runs.score_binarized == 1) & (runs.score_cont < 0.5)).sum()),
        "bin0_with_score_cont_ge_0.999": int(((runs.score_binarized == 0) & (runs.score_cont >= 0.999)).sum()),
        "fatal_error_from_by_score_binarized": {str(i): {str(c): int(v) for c, v in r.items()} for i, r in ct.iterrows()},
        "long_task_successes_by_agent": long_succ.alias.value_counts().to_dict(),
        "long_task_successes_total": int(len(long_succ)),
    }


def implied_k_fit(d: pd.DataFrame) -> float:
    """success ~ b0 + b1*log2(min) + b2*estimate. Writing b2 = b1*delta, estimate-anchored tasks
    behave like baselined tasks of length label * k*, with k* = 2**(b2/b1)
    (k* > 1: harder than their label implies, relative to baselined tasks of the same label)."""
    X = np.c_[np.log2(d.human_minutes.to_numpy()), d.human_source.eq("estimate").to_numpy(float)]
    y = d.score_binarized.to_numpy(float)
    if len(np.unique(y)) < 2:
        return np.nan
    m = LogisticRegression(C=1 / C.REGULARIZATION).fit(X, y, sample_weight=d[C.WEIGHTING].to_numpy())
    b1, b2 = m.coef_[0]
    return float(2 ** (b2 / b1)) if b1 < 0 else np.nan


def implied_k() -> pd.DataFrame:
    runs = C.load_runs().rename(columns={"alias": "agent"})
    agents = sorted(a for a in runs.agent.unique() if a != "human")
    point = {a: implied_k_fit(runs[runs.agent == a]) for a in agents}

    def boot(i: int) -> dict:
        s = bootstrap_sample(runs, ["task_family", "task_id", "run_id"], np.random.default_rng(C.SEED_BASE + i))
        return {a: implied_k_fit(s[s.agent == a]) for a in agents}

    bs = pd.DataFrame(Parallel(n_jobs=-1)(delayed(boot)(i) for i in range(C.N_BOOTSTRAP)))
    dates = C.load_release_dates()["date"]
    return pd.DataFrame([{"agent": a, "release_date": dates[a], "implied_k": point[a],
                          "ci_lo": np.nanquantile(bs[a], .025), "ci_hi": np.nanquantile(bs[a], .975)}
                         for a in agents]).sort_values("release_date")


def main() -> None:
    out = k4()
    (C.RESULTS / "k4_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "columns"}, indent=2))
    ik = implied_k()
    ik.to_csv(C.RESULTS / "implied_k_estimate_vs_baseline.csv", index=False)
    print(ik.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
