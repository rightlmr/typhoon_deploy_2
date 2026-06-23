"""Relative vorticity calculation shared by ERA5 and AIFS."""

from __future__ import annotations

import numpy as np

from tclocator.common import EARTH_RADIUS_M


def calc_vo850(u: np.ndarray, v: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    """Calculate 850 hPa relative vorticity from u/v wind components.

    Parameters
    ----------
    u, v:
        Arrays with the same shape. The final two dimensions must be
        ``(lat, lon)`` and latitudes must be ordered north-to-south.
    lat1d, lon1d:
        One-dimensional latitude and longitude coordinate arrays matching the
        final two dimensions of ``u`` and ``v``.

    Returns
    -------
    np.ndarray
        Relative vorticity ``dv/dx - du/dy`` in s^-1, as float32.
    """

    if u.shape != v.shape:
        raise ValueError(f"u and v shapes differ: {u.shape} vs {v.shape}")
    if u.shape[-2:] != (len(lat1d), len(lon1d)):
        raise ValueError(
            f"Coordinate lengths {(len(lat1d), len(lon1d))} do not match field shape {u.shape[-2:]}"
        )

    u64 = np.asarray(u, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat1d, dtype=np.float64)
    lon = np.asarray(lon1d, dtype=np.float64)

    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("At least two latitude and longitude points are required")

    dlon_rad = float(np.deg2rad(np.mean(np.abs(np.diff(lon)))))
    dlat_rad = float(np.deg2rad(np.mean(np.abs(np.diff(lat)))))
    dx = EARTH_RADIUS_M * dlon_rad * np.cos(np.deg2rad(lat))
    dx = np.maximum(dx, 1.0)
    dy = EARTH_RADIUS_M * dlat_rad

    dv_dx = np.gradient(v64, axis=-1, edge_order=1) / dx.reshape((1,) * (v64.ndim - 2) + (-1, 1))
    du_dy = np.gradient(u64, axis=-2, edge_order=1) / (-dy)
    return (dv_dx - du_dy).astype(np.float32)

