"""Decode heatmap and offset predictions into TC center detections."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from tclocator.common import DomainConfig, grid_to_latlon


def _to_numpy(array: object) -> np.ndarray:
    """Convert a NumPy or Torch tensor-like object to NumPy."""

    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()  # type: ignore[assignment]
    arr = np.asarray(array)
    return arr


def nms_peaks(heatmap: np.ndarray) -> np.ndarray:
    """Return a boolean mask of 3x3 local maxima."""

    h, w = heatmap.shape
    padded = np.pad(heatmap, ((1, 1), (1, 1)), mode="edge")
    local_max = np.full_like(heatmap, -np.inf, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            local_max = np.maximum(local_max, padded[dy : dy + h, dx : dx + w])
    return heatmap == local_max


def decode_heatmap(
    heatmap: object,
    offset: object,
    domain: DomainConfig,
    *,
    iso_time: str | None = None,
    lead_hour: int | None = None,
    conf_thresh: float = 0.3,
    lat_filter: tuple[float, float] | Iterable[float] = (0.0, 40.0),
    topk: int | None = None,
) -> pd.DataFrame:
    """Decode a single heatmap/offset pair into a detection DataFrame."""

    hm = _to_numpy(heatmap).astype(np.float32)
    off = _to_numpy(offset).astype(np.float32)
    if hm.ndim == 3:
        hm = hm[0]
    if off.ndim == 4:
        off = off[0]
    if off.shape[0] != 2:
        raise ValueError(f"Offset must have shape [2,H,W], got {off.shape}")

    keep = nms_peaks(hm) & (hm >= conf_thresh)
    ys, xs = np.where(keep)
    rows: list[dict[str, object]] = []
    lat_min, lat_max = tuple(float(v) for v in lat_filter)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        y_f = float(y) + float(off[0, y, x])
        x_f = float(x) + float(off[1, y, x])
        lat, lon = grid_to_latlon(y_f, x_f, domain)
        if not (lat_min <= float(lat) <= lat_max):
            continue
        row: dict[str, object] = {
            "ISO_TIME": iso_time,
            "LAT": float(lat),
            "LON": float(lon),
            "CONF": float(hm[y, x]),
        }
        if lead_hour is not None:
            row["LEAD_HOUR"] = int(lead_hour)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["ISO_TIME", "LAT", "LON", "CONF"] + (["LEAD_HOUR"] if lead_hour is not None else []))
    if not df.empty:
        df = df.sort_values("CONF", ascending=False).reset_index(drop=True)
        if topk is not None:
            df = df.head(topk).reset_index(drop=True)
    return df

