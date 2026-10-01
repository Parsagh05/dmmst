"""Regression: the numeric parser used METABRIC's column layout for every dataset, so on
SUPPORT five measurements were left unscaled and mean blood pressure (x7) became a
constant. Columns must follow the dataset's own modality layout."""

import json
from pathlib import Path

import numpy as np
import pytest

from sat.data.dataset.parse_metabric_numerics import metabric

REPO = Path(__file__).resolve().parents[2]
SUPPORT = REPO / "data" / "support" / "support_train_test.h5"
SUPPORT_MODALITY = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("method", ["standard", "quantile"])
def test_support_numeric_columns_are_all_scaled_and_kept(tmp_path, method):
    if not SUPPORT.is_file():
        pytest.skip("SUPPORT data not available")
    metabric(source=str(SUPPORT), processed_dir=str(tmp_path), name="s", scale_method=method,
             modality=SUPPORT_MODALITY)()
    rows = [json.loads(line) for line in open(tmp_path / "s" / "s.json")]
    num = np.array([r["numerics"] for r in rows])
    mod = np.array(SUPPORT_MODALITY)
    cont = num[:, mod == 1]
    assert np.allclose(cont.mean(0), 0.0, atol=0.05)
    assert np.allclose(cont.std(0), 1.0, atol=0.15)  # every numeric column scaled
    assert cont[:, 1].std() > 0.5  # x7 (mean blood pressure) is not a constant
    toks = rows[0]["x"].split()
    assert all(t.startswith(f"x{i}_") for i, t in enumerate(toks) if mod[i] == 0)
    assert all(t == f"x{i}" for i, t in enumerate(toks) if mod[i] == 1)
