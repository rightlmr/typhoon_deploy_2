"""Shared grid, coordinate, padding, configuration, and reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import yaml

EARTH_RADIUS_M = 6_371_000.0
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0
PHASE0_REQUIRED_MESSAGE = "请先运行 scripts/phase0_consistency_and_displacement.py 并将结论填入 config"


@dataclass(frozen=True)
class DomainConfig:
    """Regular latitude/longitude domain.

    Latitudes are represented north-to-south and longitudes west-to-east in the
    0-360 degree convention.
    """

    lat_min: float = 0.0
    lat_max: float = 70.0
    lon_min: float = 100.0
    lon_max: float = 320.0
    res: float = 0.25

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DomainConfig":
        """Build a domain from a config mapping."""

        if data is None:
            return cls()
        return cls(
            lat_min=float(data.get("lat_min", cls.lat_min)),
            lat_max=float(data.get("lat_max", cls.lat_max)),
            lon_min=float(data.get("lon_min", cls.lon_min)),
            lon_max=float(data.get("lon_max", cls.lon_max)),
            res=float(data.get("res", cls.res)),
        )

    @property
    def height(self) -> int:
        """Number of latitude rows."""

        return int(round((self.lat_max - self.lat_min) / self.res)) + 1

    @property
    def width(self) -> int:
        """Number of longitude columns."""

        return int(round((self.lon_max - self.lon_min) / self.res)) + 1

    @property
    def shape(self) -> tuple[int, int]:
        """Domain shape as ``(height, width)``."""

        return (self.height, self.width)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Save a mapping as UTF-8 YAML."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def deep_update(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` with ``overlay`` and return ``base``."""

    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return dict(base)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and Torch when available."""

    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        return


def resolve_device(device: str) -> str:
    """Resolve a config device string to ``cuda`` or ``cpu``."""

    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def build_lat_lon(domain: DomainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return domain latitude and longitude vectors."""

    lat = domain.lat_max - np.arange(domain.height, dtype=np.float64) * domain.res
    lon = domain.lon_min + np.arange(domain.width, dtype=np.float64) * domain.res
    return lat, lon


def normalize_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Normalize longitude to the 0-360 degree convention."""

    return np.mod(lon, 360.0)


def latlon_to_grid(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    domain: DomainConfig,
    *,
    clip: bool = False,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert latitude/longitude to floating grid coordinates ``(y, x)``."""

    lon360 = normalize_lon(lon)
    y = (domain.lat_max - np.asarray(lat)) / domain.res
    x = (np.asarray(lon360) - domain.lon_min) / domain.res
    if clip:
        y = np.clip(y, 0.0, domain.height - 1.0)
        x = np.clip(x, 0.0, domain.width - 1.0)
    if np.isscalar(lat) and np.isscalar(lon):
        return float(y), float(x)
    return y, x


def grid_to_latlon(
    y: np.ndarray | float,
    x: np.ndarray | float,
    domain: DomainConfig,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert floating grid coordinates ``(y, x)`` to latitude/longitude."""

    lat = domain.lat_max - np.asarray(y) * domain.res
    lon = domain.lon_min + np.asarray(x) * domain.res
    lon = normalize_lon(lon)
    if np.isscalar(y) and np.isscalar(x):
        return float(lat), float(lon)
    return lat, lon


def in_domain(lat: float, lon: float, domain: DomainConfig) -> bool:
    """Return whether a lat/lon position is inside the configured domain."""

    lon360 = float(normalize_lon(lon))
    return domain.lat_min <= lat <= domain.lat_max and domain.lon_min <= lon360 <= domain.lon_max


def pad_hw_to_multiple(
    height: int,
    width: int,
    multiple: int = 32,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Return padded shape and ``(top, bottom, left, right)`` padding."""

    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return height + pad_h, width + pad_w, (0, pad_h, 0, pad_w)


def pad_array_to_multiple(
    array: np.ndarray,
    *,
    multiple: int = 32,
    mode: str = "edge",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad the last two dimensions of an array to a multiple of ``multiple``."""

    height, width = array.shape[-2:]
    _, _, pads = pad_hw_to_multiple(height, width, multiple)
    top, bottom, left, right = pads
    pad_width = [(0, 0)] * array.ndim
    pad_width[-2] = (top, bottom)
    pad_width[-1] = (left, right)
    if bottom == 0 and right == 0:
        return array, pads
    return np.pad(array, pad_width, mode=mode), pads


def crop_array_to_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop the last two dimensions of an array to ``shape``."""

    height, width = shape
    return array[..., :height, :width]


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in kilometers."""

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    if np.isscalar(lat1) and np.isscalar(lat2):
        return float(dist)
    return dist


def nearest_indices(source: Sequence[float], target: Sequence[float], *, tolerance: float | None = None) -> np.ndarray:
    """Find nearest source indices for each target coordinate."""

    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    indices = np.empty(target_arr.shape, dtype=np.int64)
    for i, value in enumerate(target_arr):
        idx = int(np.argmin(np.abs(source_arr - value)))
        if tolerance is not None and abs(float(source_arr[idx]) - float(value)) > tolerance:
            raise ValueError(f"No coordinate within tolerance for {value}; nearest is {source_arr[idx]}")
        indices[i] = idx
    return indices


def crop_regular_latlon_grid(
    values: np.ndarray,
    source_lat: Sequence[float],
    source_lon: Sequence[float],
    domain: DomainConfig,
) -> np.ndarray:
    """Crop/reindex a 2D regular lat/lon grid to the configured domain."""

    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {values.shape}")
    target_lat, target_lon = build_lat_lon(domain)
    source_lon360 = np.asarray(normalize_lon(np.asarray(source_lon, dtype=np.float64)))
    lat_idx = nearest_indices(source_lat, target_lat, tolerance=domain.res * 0.51)
    lon_idx = nearest_indices(source_lon360, target_lon, tolerance=domain.res * 0.51)
    cropped = values[np.ix_(lat_idx, lon_idx)]
    if cropped.shape != domain.shape:
        raise ValueError(f"Cropped shape {cropped.shape} != expected {domain.shape}")
    return cropped.astype(np.float32, copy=False)


def iter_files(directory: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """Return sorted files under ``directory`` with one of ``suffixes``."""

    root = Path(directory)
    if not root.exists():
        return []
    suffix_set = {s.lower() for s in suffixes}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffix_set)


def require_config_value(config: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Read a nested config value and raise ``KeyError`` if missing."""

    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current

