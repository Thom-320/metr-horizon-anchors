#!/usr/bin/env bash
# Fetch pinned public inputs and build the pinned venv. Public data only; nothing is pushed anywhere.
set -euo pipefail
cd "$(dirname "$0")"
METR_COMMIT=52cb829c7a2efb2d659285c4b1768d191d97f8d2   # METR/eval-analysis-public main @ 2026-03-05 "Sync public pipeline (#39)"
RUNS_SHA256=609f904f4b6ae32129388da89d036e00bac511ad94223d2be1e58fc2b45b55cd  # reports/time-horizon-1-1/data/raw/runs.jsonl
PUB_SHA256=aae31902b0519a4da73e16643915e5e8aca13cd3315c3aac893ce3d6dfe92ad9   # metr.org/assets/benchmark_results_1_1.yaml as fetched 2026-10-08
if [ ! -d vendor/eval-analysis-public/.git ]; then
  mkdir -p vendor && git clone -q https://github.com/METR/eval-analysis-public.git vendor/eval-analysis-public
fi
git -C vendor/eval-analysis-public checkout -q "$METR_COMMIT"
echo "$RUNS_SHA256  vendor/eval-analysis-public/reports/time-horizon-1-1/data/raw/runs.jsonl" | shasum -a 256 -c -
mkdir -p published
[ -f published/benchmark_results_1_1.yaml ] || curl -sL https://metr.org/assets/benchmark_results_1_1.yaml -o published/benchmark_results_1_1.yaml
echo "$PUB_SHA256  published/benchmark_results_1_1.yaml" | shasum -a 256 -c - || \
  echo "WARNING: metr.org file changed since 2026-10-08; the gate compares against whatever is there now"
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python -r requirements.lock.txt
  uv pip install -q --python .venv/bin/python --no-deps -e vendor/eval-analysis-public
fi
.venv/bin/python -c "import sys, sklearn, numpy, pandas; print(sys.version.split()[0], 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"
