"""TC Locator package.

The package implements a CenterNet-style tropical cyclone center locator from
ERA5/AIFS meteorological fields.
"""

import warnings

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.*")
        import pygrib as _pygrib  # noqa: F401
except ImportError:
    _pygrib = None  # type: ignore[assignment]

__all__ = [
    "common",
    "io_era5",
    "io_aifs",
    "vorticity",
    "normalization",
    "labels",
    "dataset",
    "model",
    "losses",
    "decode",
    "tracking",
    "metrics",
]
