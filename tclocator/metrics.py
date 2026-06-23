"""Evaluation metrics for field-consistent TC localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from tclocator.common import haversine_km


@dataclass(frozen=True)
class LeadBin:
    """A forecast lead bin."""

    name: str
    min_hour: int
    max_hour: int


DEFAULT_LEAD_BINS = (
    LeadBin("000-024", 0, 24),
    LeadBin("024-048", 24, 48),
    LeadBin("048-096", 48, 96),
    LeadBin("096-120", 96, 120),
)


def match_predictions(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    *,
    hit_threshold_km: float = 60.0,
) -> pd.DataFrame:
    """Greedily match predictions to references for each comparable forecast field.

    AIFS files can share the same valid time across different initialization
    times and forecast leads. When both inputs carry ``LEAD_HOUR``, it is part
    of the matching key so a prediction from one forecast field cannot satisfy a
    reference from another field with the same ``ISO_TIME``.
    """

    rows: list[dict[str, object]] = []
    if references.empty:
        return pd.DataFrame(rows)

    pred = predictions.copy()
    ref = references.copy()
    pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"], utc=True, errors="coerce")
    ref["ISO_TIME"] = pd.to_datetime(ref["ISO_TIME"], utc=True, errors="coerce")
    group_keys = ["ISO_TIME"]
    if "LEAD_HOUR" in pred.columns and "LEAD_HOUR" in ref.columns:
        pred["LEAD_HOUR"] = pd.to_numeric(pred["LEAD_HOUR"], errors="coerce").astype("Int64")
        ref["LEAD_HOUR"] = pd.to_numeric(ref["LEAD_HOUR"], errors="coerce").astype("Int64")
        group_keys.append("LEAD_HOUR")

    for key, ref_group in ref.groupby(group_keys):
        if len(group_keys) == 1:
            time = key
            pred_group = pred.loc[pred["ISO_TIME"] == time].copy()
        else:
            time, lead_hour = key
            pred_group = pred.loc[(pred["ISO_TIME"] == time) & (pred["LEAD_HOUR"] == lead_hour)].copy()
        used_pred: set[int] = set()
        for _, ref_row in ref_group.iterrows():
            best_idx: int | None = None
            best_dist = float("inf")
            for pred_idx, pred_row in pred_group.iterrows():
                if int(pred_idx) in used_pred:
                    continue
                dist = haversine_km(
                    float(pred_row["LAT"]),
                    float(pred_row["LON"]),
                    float(ref_row["LAT_FIELD"]),
                    float(ref_row["LON_FIELD"]),
                )
                if dist < best_dist:
                    best_dist = float(dist)
                    best_idx = int(pred_idx)
            pred_row = pred_group.loc[best_idx] if best_idx is not None else None
            if best_idx is not None:
                used_pred.add(best_idx)
            loc_error = best_dist if best_idx is not None else np.nan
            track_bias = haversine_km(
                float(ref_row["LAT_FIELD"]),
                float(ref_row["LON_FIELD"]),
                float(ref_row["LAT_TRUE"]),
                float(ref_row["LON_TRUE"]),
            )
            end2end = (
                haversine_km(
                    float(pred_row["LAT"]),
                    float(pred_row["LON"]),
                    float(ref_row["LAT_TRUE"]),
                    float(ref_row["LON_TRUE"]),
                )
                if pred_row is not None
                else np.nan
            )
            rows.append(
                {
                    "ISO_TIME": time,
                    "SID": ref_row.get("SID"),
                    "LEAD_HOUR": int(ref_row.get("LEAD_HOUR", -1)),
                    "loc_error_km": loc_error,
                    "track_bias_km": float(track_bias),
                    "end2end_km": float(end2end) if not np.isnan(end2end) else np.nan,
                    "hit": bool(not np.isnan(loc_error) and loc_error <= hit_threshold_km),
                    "matched": best_idx is not None,
                }
            )
    return pd.DataFrame(rows)


def summarize_by_lead(
    matched: pd.DataFrame,
    *,
    lead_bins: Iterable[LeadBin] = DEFAULT_LEAD_BINS,
) -> pd.DataFrame:
    """Summarize matched metrics by forecast lead bins."""

    rows: list[dict[str, object]] = []
    if matched.empty:
        return pd.DataFrame(rows)

    for lead_bin in lead_bins:
        part = matched.loc[(matched["LEAD_HOUR"] >= lead_bin.min_hour) & (matched["LEAD_HOUR"] < lead_bin.max_hour)]
        if part.empty:
            continue
        rows.append(
            {
                "lead_bin": lead_bin.name,
                "n_ref": int(len(part)),
                "recall": float(part["hit"].mean()),
                "loc_error_median_km": float(part["loc_error_km"].median(skipna=True)),
                "track_bias_median_km": float(part["track_bias_km"].median(skipna=True)),
                "end2end_median_km": float(part["end2end_km"].median(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def false_alarm_rate(predictions: pd.DataFrame, matched: pd.DataFrame) -> float:
    """Estimate false alarm rate as unmatched predictions / predictions."""

    if predictions.empty:
        return 0.0
    matched_count = int(matched["matched"].sum()) if not matched.empty and "matched" in matched else 0
    return max(0.0, float(len(predictions) - matched_count) / float(len(predictions)))


def precision_recall_curve(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    hit_threshold_km: float = 60.0,
) -> pd.DataFrame:
    """Calculate PR points across confidence thresholds."""

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = predictions.loc[predictions["CONF"] >= threshold]
        matched = match_predictions(pred, references, hit_threshold_km=hit_threshold_km)
        tp = int(matched["hit"].sum()) if not matched.empty else 0
        fp = max(0, len(pred) - tp)
        fn = max(0, len(references) - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({"conf_thresh": float(threshold), "precision": precision, "recall": recall})
    return pd.DataFrame(rows)
