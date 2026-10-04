"""Generic multi-event dataset parser.

Reads one CSV with the layout

    id, <feature columns>, event1..eventK, duration1..durationK

and writes the SAT json-lines representation (``x`` / ``modality`` / ``numerics`` /
``events`` / ``durations``) that every other parser produces. It covers any number of
events and any regime (competing, semi-competing, non-competing): the regime lives in
how the CSV was built, not in the parser.

Feature columns are picked by prefix unless given explicitly:

* ``x_*`` - numeric. Standardised; the token is the feature name and the value is fed
  through the numeric embedding (paper Sec. 2.2, Fig. 2).
* ``c_*`` - categorical. The token is ``<name>_<value>`` (e.g. ``c_agecl_gt40``) with a
  numeric value of 1, i.e. the paper's categorical token encoding.

Used by the synthetic simulator (``sat.data.simulate``) and by EBMT
(``scripts/prepare_ebmt.py``).
"""

__authors__ = ["Parsa"]
__status__ = "Development"

import re
from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path
from typing import List, Optional

import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sat.utils import logging

logger = logging.get_default_logger()


_SYMBOLS = [(">=", "ge"), ("<=", "le"), (">", "gt"), ("<", "lt"), ("=", "eq"), ("+", "plus")]


def _clean(value) -> str:
    """Categorical values become part of a whitespace-split token: no spaces allowed.
    Comparison symbols are spelled out so that e.g. '>40' and '<=20' stay distinct
    and readable ('gt40', 'le20')."""
    v = str(value)
    for sym, word in _SYMBOLS:
        v = v.replace(sym, word)
    return re.sub(r"[^0-9A-Za-z]+", "_", v).strip("_") or "NA"


@dataclass
class multievent:
    source: str
    processed_dir: str
    name: str
    num_events: int
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    scale_method: str = "standard"
    scale_numerics: bool = True
    min_scale_numerics: float = 1.0

    @log_on_start(DEBUG, "Create multi-event data representation...")
    @log_on_error(
        ERROR,
        "Error creating multi-event data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
        df = pd.read_csv(self.source)

        numeric = list(self.numeric_features or [c for c in df.columns if c.startswith("x_")])
        categorical = list(
            self.categorical_features or [c for c in df.columns if c.startswith("c_")]
        )
        if not numeric and not categorical:
            raise ValueError(f"No feature columns (x_* / c_*) found in {self.source}")

        event_cols = [f"event{k + 1}" for k in range(self.num_events)]
        duration_cols = [f"duration{k + 1}" for k in range(self.num_events)]
        missing = [c for c in event_cols + duration_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{self.source} lacks columns {missing} for num_events={self.num_events}")

        num = df[numeric].astype(float)
        if self.scale_numerics and numeric:
            if self.scale_method == "standard":
                num = pd.DataFrame(StandardScaler().fit_transform(num), columns=numeric)
            elif self.scale_method == "min_max":
                num = pd.DataFrame(
                    MinMaxScaler().fit_transform(num) + self.min_scale_numerics,
                    columns=numeric,
                )
            else:
                raise ValueError(
                    f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'"
                )

        cat = df[categorical].map(_clean) if categorical else pd.DataFrame(index=df.index)

        # categorical first, then numeric: modality 0 = token, 1 = numeric value
        modality = [0] * len(categorical) + [1] * len(numeric)
        x, numerics = [], []
        for i in range(len(df)):
            toks = [f"{c}_{cat.iat[i, j]}" for j, c in enumerate(categorical)] + numeric
            vals = [1.0] * len(categorical) + [float(v) for v in num.iloc[i].values]
            x.append(" ".join(toks))
            numerics.append(vals)

        data = pd.DataFrame(
            {
                "id": df["id"].values,
                "x": x,
                "modality": [modality] * len(df),
                "numerics": numerics,
                "events": df[event_cols].astype(int).values.tolist(),
                "durations": df[duration_cols].astype(float).values.tolist(),
            }
        )

        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        data.to_json(Path(f"{out_dir}/{self.name}.json"), orient="records", lines=True)
        logger.info(
            f"Wrote {len(data)} records, {len(categorical)} categorical + {len(numeric)} "
            f"numeric features, {self.num_events} events -> {out_dir}"
        )
