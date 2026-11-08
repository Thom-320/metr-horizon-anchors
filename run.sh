#!/usr/bin/env bash
# Reproduce every number in PILOT-MEMO.md. ~5 min on a 10-core laptop, CPU only, no network after setup.
set -euo pipefail
cd "$(dirname "$0")"
./setup.sh
PY=".venv/bin/python -I"
$PY scripts/01_replicate.py         > results/log_01_replicate.txt 2>&1   # K1 gate + doubling time
$PY scripts/02_anchor.py            > results/log_02_anchor.txt 2>&1      # anchor split, k sensitivity, K2
$PY scripts/03_k4_and_diagnostics.py > results/log_03_k4.txt 2>&1         # K4 encoding check, implied k
$PY scripts/04_figures.py           > results/log_04_figures.txt 2>&1
cat results/replication_summary.json results/k2_summary.json
