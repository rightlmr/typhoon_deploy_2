"""ERA5 NetCDF reading, domain cropping, and channel stacking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from tclocator.common import DomainConfig, build_lat_lon, crop_regular_latlon_grid
from tclocator.vorticity import calc_vo850


LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")


def _coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    """Find a coordinate name from candidate names."""

    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"None of the coordinate names exist: {candidates}")


def _squeeze_to_2d(da: xr.DataArray) -> np.ndarray:
    """Squeeze singleton dimensions and return a 2D array."""

    squeezed = da.squeeze(drop=True)
    if squeezed.ndim > 2:
        indexers = {dim: 0 for dim in squeezed.dims[:-2]}
        squeezed = squeezed.isel(indexers)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected 2D variable after squeeze, got dims {squeezed.dims}")
    return np.asarray(squeezed.values, dtype=np.float32)


def _read_var(ds: xr.Dataset, var_name: str) -> np.ndarray:
    """Read a 2D variable from an xarray dataset."""

    if var_name not in ds:
        raise KeyError(f"ERA5 variable not found in dataset: {var_name}")
    return _squeeze_to_2d(ds[var_name])


def read_era5_channels(
    path: str | Path,
    *,
    channels: Sequence[str],
    domain: DomainConfig,
    era5_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a configured ERA5 field as ``[C,H,W]`` float32."""

    var_map = dict(era5_config.get("var_map", {}))
    vo850_from_uv = bool(era5_config.get("vo850_from_uv", True))
    arrays: dict[str, np.ndarray] = {}

    with xr.open_dataset(path) as ds:
        lat_name = _coord_name(ds, LAT_NAMES)
        lon_name = _coord_name(ds, LON_NAMES)
        source_lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        source_lon = np.asarray(ds[lon_name].values, dtype=np.float64)

        def read_internal(internal_name: str) -> np.ndarray:
            mapped = var_map.get(internal_name, internal_name)
            return crop_regular_latlon_grid(_read_var(ds, mapped), source_lat, source_lon, domain)

        for channel in channels:
            if channel == "vo_850":
                if vo850_from_uv:
                    u = arrays.get("u850")
                    if u is None:
                        u = read_internal("u850")
                        arrays["u850"] = u
                    v = arrays.get("v850")
                    if v is None:
                        v = read_internal("v850")
                        arrays["v850"] = v
                    lat1d, lon1d = build_lat_lon(domain)
                    arrays[channel] = calc_vo850(u, v, lat1d, lon1d)
                else:
                    arrays[channel] = read_internal("vo_850")
            else:
                arrays[channel] = read_internal(channel)

    stacked = np.stack([arrays[channel] for channel in channels], axis=0).astype(np.float32)
    meta = {"path": str(path)}
    return stacked, meta


def read_era5_msl(
    path: str | Path,
    *,
    domain: DomainConfig,
    era5_config: Mapping[str, Any],
) -> np.ndarray:
    """Read only sea-level pressure from an ERA5 file."""

    var_map = dict(era5_config.get("var_map", {}))
    msl_name = var_map.get("msl", "msl")
    with xr.open_dataset(path) as ds:
        lat_name = _coord_name(ds, LAT_NAMES)
        lon_name = _coord_name(ds, LON_NAMES)
        return crop_regular_latlon_grid(
            _read_var(ds, msl_name),
            np.asarray(ds[lat_name].values, dtype=np.float64),
            np.asarray(ds[lon_name].values, dtype=np.float64),
            domain,
        )

