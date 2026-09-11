"""Covariate drift: PSI of model features vs a training baseline.

Bins / keep-lists are frozen at period 0. Re-cutting quantiles on each batch
would hide the same shift that quantile High hides on the call list.

intent_score PSI uses *observed* values only. Missingness is intent_drift.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from agent.config import DEFAULT_FEATURE_COLUMNS
from monitoring.config import PSI_FLOOR, PSI_N_BINS, PSI_RARE_CATEGORY

CATEGORICAL_FEATURES = ("account_type", "industry")
# Missing intent is a coverage gap, already watched. Filling 25.3 before PSI
# would make the vendor hole look like "typical intent."
OBSERVED_ONLY = frozenset({"intent_score"})
OTHER = "__other__"
MISSING = "__missing__"


def _numeric_features() -> tuple[str, ...]:
    cats = set(CATEGORICAL_FEATURES)
    return tuple(c for c in DEFAULT_FEATURE_COLUMNS if c not in cats)


def _share_map(keys: list[str], counts: pd.Series) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: float(counts.get(k, 0)) / total for k in keys}


def _cat_key(val: Any, keep: set[str]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
        return MISSING
    s = str(val)
    return s if s in keep else OTHER


def _disc_key(val: Any, values: set[str]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
        return MISSING
    key = _num_token(val)
    return key if key in values else OTHER


def _num_token(val: Any) -> str:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isfinite(f) and f == int(f):
        return str(int(f))
    return str(f)


def _dump_edge(x: float) -> float | str:
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return float(x)


def _load_edge(x: float | str) -> float:
    if x == "-inf":
        return -math.inf
    if x == "inf":
        return math.inf
    return float(x)


def _pretty_edge(x: float) -> str:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == int(x):
        return str(int(x))
    return f"{x:.1f}"


def _quantile_spec(series: pd.Series, n_bins: int = PSI_N_BINS) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    nunique = int(s.nunique())
    if nunique <= n_bins:
        values = sorted({_num_token(v) for v in s.tolist()})
        keys = list(values)
        mapped = s.map(lambda v: _disc_key(v, set(values)))
        # Train should not need OTHER; include it only if it appeared.
        if (mapped == OTHER).any():
            keys.append(OTHER)
        counts = mapped.value_counts()
        return {
            "kind": "discrete",
            "values": values,
            "keys": keys,
            "train_shares": _share_map(keys, counts),
        }

    qs = [i / n_bins for i in range(n_bins + 1)]
    edges = pd.Series(s.quantile(qs)).to_numpy(dtype=float)
    edges = pd.unique(edges)
    if len(edges) < 3:
        return _quantile_spec(s, n_bins=max(nunique, 1))
    edges[0] = -math.inf
    edges[-1] = math.inf
    cut = pd.cut(s, bins=edges, include_lowest=True)
    keys = [str(i) for i in range(len(edges) - 1)]
    codes = cut.cat.codes
    mapped = pd.Series([str(c) if c >= 0 else MISSING for c in codes], index=s.index)
    counts = mapped.value_counts()
    return {
        "kind": "quantile",
        "edges": [_dump_edge(float(e)) for e in edges],
        "keys": keys,
        "train_shares": _share_map(keys, counts),
    }


def _categorical_spec(series: pd.Series) -> dict[str, Any]:
    work = series.where(series.notna(), other=pd.NA)
    as_str = work.map(lambda v: MISSING if pd.isna(v) else str(v))
    shares = as_str.value_counts(normalize=True)
    keep = sorted(k for k, p in shares.items() if k != MISSING and p >= PSI_RARE_CATEGORY)
    keep_set = set(keep)
    mapped = as_str.map(lambda v: v if v == MISSING or v in keep_set else OTHER)
    keys = list(keep)
    if (mapped == OTHER).any():
        keys.append(OTHER)
    if (mapped == MISSING).any():
        keys.append(MISSING)
    counts = mapped.value_counts()
    return {
        "kind": "categorical",
        "keep": keep,
        "keys": keys,
        "train_shares": _share_map(keys, counts),
    }


def build_feature_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Period-0 bin edges and train mass. Call on training_data only."""
    specs: dict[str, Any] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        specs[col] = _categorical_spec(df[col])
    for col in _numeric_features():
        if col not in df.columns:
            continue
        series = df[col]
        if col in OBSERVED_ONLY:
            series = series.dropna()
        specs[col] = _quantile_spec(series)
        if col in OBSERVED_ONLY:
            specs[col]["observed_only"] = True
    return specs


def _batch_shares(series: pd.Series, spec: dict[str, Any]) -> dict[str, float]:
    kind = spec["kind"]
    keys = list(spec.get("keys") or [])
    if kind == "categorical":
        keep = set(spec.get("keep") or [])
        mapped = series.map(lambda v: _cat_key(v, keep))
    elif kind == "discrete":
        values = set(spec.get("values") or [])
        mapped = pd.to_numeric(series, errors="coerce").map(
            lambda v: _disc_key(v, values)
        )
    else:
        edges = [_load_edge(e) for e in spec.get("edges") or []]
        num = pd.to_numeric(series, errors="coerce")
        cut = pd.cut(num.dropna(), bins=edges, include_lowest=True)
        mapped = pd.Series(MISSING, index=series.index, dtype=object)
        mapped.loc[num.dropna().index] = [str(c) if c >= 0 else MISSING for c in cut.cat.codes]
        still_na = num.isna() & series.notna()
        mapped.loc[still_na] = OTHER
        if spec.get("observed_only"):
            mapped = mapped.loc[num.notna()]
        if MISSING not in keys and (mapped == MISSING).any():
            keys.append(MISSING)
        if OTHER not in keys and (mapped == OTHER).any():
            keys.append(OTHER)
        counts = mapped.value_counts()
        all_keys = list(dict.fromkeys([*keys, *counts.index.astype(str)]))
        return _share_map(all_keys, counts)

    if MISSING not in keys and (mapped == MISSING).any():
        keys.append(MISSING)
    if OTHER not in keys and (mapped == OTHER).any():
        keys.append(OTHER)
    counts = mapped.value_counts()
    all_keys = list(dict.fromkeys([*keys, *counts.index.astype(str)]))
    return _share_map(all_keys, counts)


def _psi_pair(p: float, q: float, floor: float = PSI_FLOOR) -> float:
    p = max(p, floor)
    q = max(q, floor)
    return (p - q) * math.log(p / q)


def _bin_label(spec: dict[str, Any], key: str) -> str:
    if key in {OTHER, MISSING}:
        return key
    kind = spec["kind"]
    if kind == "quantile":
        try:
            i = int(key)
        except (TypeError, ValueError):
            return key
        edges = [_load_edge(e) for e in spec.get("edges") or []]
        if 0 <= i < len(edges) - 1:
            return f"[{_pretty_edge(edges[i])}, {_pretty_edge(edges[i + 1])})"
        return key
    return key


def _top_move(spec: dict[str, Any], batch: dict[str, float], train: dict[str, float]) -> str:
    keys = set(batch) | set(train)
    if not keys:
        return "—"
    key = max(keys, key=lambda k: abs(batch.get(k, 0.0) - train.get(k, 0.0)))
    b, t = batch.get(key, 0.0), train.get(key, 0.0)
    if abs(b - t) < 1e-12:
        return "no shift"
    return f"{_bin_label(spec, key)} {100 * t:.1f}% -> {100 * b:.1f}%"


def feature_psi_table(df: pd.DataFrame, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in baseline.items():
        if name not in df.columns:
            continue
        series = df[name]
        if spec.get("observed_only"):
            series = series.dropna()
            if series.empty:
                continue
        train = {str(k): float(v) for k, v in (spec.get("train_shares") or {}).items()}
        batch = _batch_shares(series, spec)
        keys = set(train) | set(batch)
        psi = sum(_psi_pair(batch.get(k, 0.0), train.get(k, 0.0)) for k in keys)
        rows.append(
            {
                "feature": name,
                "kind": spec.get("kind"),
                "psi": float(psi),
                "top_move": _top_move(spec, batch, train),
            }
        )
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows
