# Reproduction

- Install Python 3.12 and `uv`, then run `./setup.sh`. This fetches public METR inputs and installs the pinned environment; no API credentials are needed. Linux needs `shasum` (provided by Perl on Ubuntu).
- Install tests separately: `uv pip install --python .venv/bin/python pytest`; run `.venv/bin/python -m pytest -q`. There were no tests before this reproduction harness.
- Offline audit: `bash scripts/reproduce_offline.sh`. Requires only Python's standard library, no vendor checkout or venv. It reads committed summaries, not raw runs, and prints sources and rounding checks. Exit 0 means all match, 1 means discrepancies or non-derivable claims, 2 means invalid/missing inputs.
- Known audit limitations: results do not record the METR commit, DVC cross-check, or longest measured task. The measured-only Opus upper CI is finite in the committed CSV, contrary to README's “unbounded” claim. The audit intentionally fails rather than conceal these issues; CI will report them too.
- Do not modify analysis code (`src/`, `scripts/0*.py`), `results/`, `figures/`, `published/`, data, README numbers, `setup.sh`, or `run.sh`. Never run `./run.sh` in this repository: it overwrites results. Do not make model API calls or use secrets. Record non-reproducing numbers with their command and difference; do not fix them. Stop and report after the same error repeats three times.
