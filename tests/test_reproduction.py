import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("reproduce_offline", ROOT / "scripts" / "reproduce_offline.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


def test_mhv_common_published_model_count():
    import mhv_common

    summary = json.loads((mhv_common.RESULTS / "replication_summary.json").read_text())
    assert summary["n_models_compared"] == 20


@pytest.mark.parametrize("value,expected,match", [("128.7438", "128.7", True), ("128.75", "128.7", True), ("128.751", "128.7", False), ("Infinity", "0.0", False)])
def test_rounding(value, expected, match):
    assert offline.within_rounding(value, expected) is match


def test_saved_figures():
    rows = offline.figures()
    assert next(r[2] for r in rows if r[0] == "24,008 runs") == "yes"
    assert next(r[2] for r in rows if r[0] == "Opus measured-only upper CI unbounded") == "NO"


def test_missing_inputs(tmp_path, capsys):
    assert offline.main(tmp_path) == 2
    assert "input error" in capsys.readouterr().err


def test_empty_input(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "weight_shares.csv").write_text("n_runs\n")
    with pytest.raises(ValueError, match="empty"):
        offline.csv_rows(tmp_path, "weight_shares.csv")
