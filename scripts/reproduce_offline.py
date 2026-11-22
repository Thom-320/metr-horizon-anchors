"""Audit README Findings against committed summaries, without fitting or fetching data."""
import csv
import math
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(root, name):
    with (root / "results" / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty results/{name}")
    return rows


def within_rounding(value, published):
    actual = Decimal(str(value))
    target = Decimal(published)
    return actual.is_finite() and abs(actual - target) <= Decimal("0.5") * Decimal(10) ** target.as_tuple().exponent


def figures(root=ROOT):
    rows = []

    def number(label, value, expected, source):
        rows.append((label, str(value), "yes" if within_rounding(value, expected) else "NO", source))

    def unknown(label, source):
        rows.append((label, "not derivable", "NOT DERIVABLE", source))

    unknown("METR commit 52cb829", "setup.sh pins it; results/ and published/ contain no provenance record")
    weights = csv_rows(root, "weight_shares.csv")
    number("24,008 runs", sum(int(r["n_runs"]) for r in weights), "24008", "results/weight_shares.csv: sum(n_runs)")
    counts = csv_rows(root, "anchor_counts_by_length.csv")
    measured = sum(int(r["baseline"]) for r in counts)
    estimated = sum(int(r["estimate"]) for r in counts)
    number("228 tasks", measured + estimated, "228", "results/anchor_counts_by_length.csv: sum(baseline + estimate)")
    gate = csv_rows(root, "replication_gate.csv")
    number("20 models", len({r["agent"] for r in gate}), "20", "results/replication_gate.csv: unique agent")
    differences = [abs(Decimal(r["p50_reproduced"]) / Decimal(r["p50_published"]) - 1) for r in gate]
    number("p50 relative difference 0.000000", max(differences), "0.000000", "results/replication_gate.csv: max abs(reproduced/published - 1); stored precision")
    doubling = csv_rows(root, "doubling_by_variant.csv")
    all_fit = next(r for r in doubling if r["variant"] == "all" and r["sota_rule"] == "variant_sota")
    base_fit = next(r for r in doubling if r["variant"] == "baseline_only" and r["sota_rule"] == "variant_sota")
    number("doubling 128.7 days", all_fit["doubling_days"], "128.7", "results/doubling_by_variant.csv: all, variant_sota, doubling_days")
    unknown("METR DVC pipeline same fits", "no committed DVC output/cross-check log")
    number("162 measured tasks", measured, "162", "results/anchor_counts_by_length.csv: sum(baseline)")
    number("66 estimated tasks", estimated, "66", "results/anchor_counts_by_length.csv: sum(estimate)")
    long_rows = [r for r in counts if r["length_bucket"] in ("8-16h", ">=16h")]
    long_measured = sum(int(r["baseline"]) for r in long_rows)
    long_estimated = sum(int(r["estimate"]) for r in long_rows)
    source = "results/anchor_counts_by_length.csv: 8-16h and >=16h buckets"
    number("31 tasks >=8h", long_measured + long_estimated, "31", source)
    number("5 measured tasks >=8h", long_measured, "5", source + ", baseline")
    number("26 estimated tasks >=8h", long_estimated, "26", source + ", estimate")
    unknown("longest measured task 18.5 h", "task-level times absent from committed results/ and published/")
    horizons = csv_rows(root, "horizons_by_variant.csv")
    all_horizons = {r["agent"]: float(r["p50"]) for r in horizons if r["variant"] == "all"}
    base_horizons = {r["agent"]: float(r["p50"]) for r in horizons if r["variant"] == "baseline_only"}
    rises = bool(all_horizons) and all_horizons.keys() == base_horizons.keys() and all(base_horizons[a] > p for a, p in all_horizons.items())
    rows.append(("every model's p50 rises", str(rises), "yes" if rises else "NO", "results/horizons_by_variant.csv: baseline_only vs all, p50"))
    opus = "Claude Opus 4.6 (Inspect)"
    for variant, expected in (("all", "12.0"), ("baseline_only", "48.6")):
        row = next(r for r in horizons if r["variant"] == variant and r["agent"] == opus)
        number(f"Opus p50 {variant} {expected} h", float(row["p50"]) / 60, expected, f"results/horizons_by_variant.csv: {variant}, Opus 4.6, p50 / 60")
    upper = float(next(r["p50_hi"] for r in horizons if r["variant"] == "baseline_only" and r["agent"] == opus)) / 60
    rows.append(("Opus measured-only upper CI unbounded", f"{upper} h", "yes" if upper == math.inf else "NO", "results/horizons_by_variant.csv: baseline_only, Opus 4.6, p50_hi / 60"))
    for column, expected in (("ratio_vs_all", "0.88"), ("ratio_lo", "0.50"), ("ratio_hi", "1.15")):
        number(f"paired doubling {column} {expected}", base_fit[column], expected, f"results/doubling_by_variant.csv: baseline_only, variant_sota, {column} (paired 95% CI)")
    return rows


def main(root=ROOT):
    print("README figure | recomputed | match | source")
    try:
        rows = figures(root)
    except (OSError, ValueError, KeyError, StopIteration, ArithmeticError) as exc:
        print(f"input error | {exc} | NO | committed results required", file=sys.stderr)
        return 2
    for row in rows:
        print(" | ".join(row))
    print("Audit only: saved summaries, not a refit or independent validation of their provenance.")
    return int(any(row[2] != "yes" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
