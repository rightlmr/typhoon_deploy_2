"""Adapter for the heatmap TC locator used by the deployment pipeline."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.*")
        import pygrib  # noqa: F401
except ImportError:
    pygrib = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
import torch

from aifs_data_utils import parse_aifs_filename
from tclocator.common import (
    DomainConfig,
    build_lat_lon,
    haversine_km,
    latlon_to_grid,
    resolve_device,
)
from tclocator.decode import decode_heatmap
from tclocator.io_aifs import read_aifs_channels
from tclocator.model import build_model_from_config
from tclocator.normalization import apply_norm, load_norm_stats


class HeatmapDetectorAdapter:
    """Expose the new heatmap locator through the original deployment contract."""

    OUTPUT_COLUMNS = ["ISO_TIME", "LAT", "LON", "MSL_MIN", "WS", "CONF"]

    def __init__(self, model_dir: str, device: str = "auto", config: Mapping[str, Any] | None = None):
        self.model_dir = self._resolve_path(model_dir)
        self.config: dict[str, Any] = dict(config or {})
        self.device = resolve_device(device)

        self.channels = list(self.config.get("channels", ["msl", "vo_850", "t_500"]))
        self.domain = DomainConfig.from_mapping(self.config.get("domain"))
        self.lat1d, self.lon1d = build_lat_lon(self.domain)
        self.decode_config = dict(self.config.get("decode", {}))
        self.msl_min_radius_km = float(self.config.get("msl_min_radius_km", 100.0))
        self.aifs_config = dict(self.config.get("aifs", {}))

        norm_stats = self.config.get("norm_stats", self.model_dir / "norm_stats_aifs.json")
        self.norm_stats_path = self._resolve_path(norm_stats)
        self.norm_stats = load_norm_stats(self.norm_stats_path)

        self.checkpoint_path = self._find_checkpoint()
        self.model = self._load_model()

        print(f"[Detection] Heatmap locator loaded: {self.checkpoint_path}")
        print(f"[Detection] Device: {self.device}")
        print(f"[Detection] Channels: {self.channels}")
        print(f"[Detection] Domain: {self.domain.shape} lat/lon grid")

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=HeatmapDetectorAdapter.OUTPUT_COLUMNS)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (_PROJECT_ROOT / path).resolve()

    def _find_checkpoint(self) -> Path:
        configured = self.config.get("checkpoint")
        if configured:
            path = self._resolve_path(configured)
            if not path.exists():
                raise FileNotFoundError(f"Detection checkpoint not found: {path}")
            return path

        preferred = [self.model_dir / "finetune_best.ckpt", self.model_dir / "last.ckpt"]
        for path in preferred:
            if path.exists():
                return path

        candidates = sorted(self.model_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No detection checkpoint found in {self.model_dir}")
        return candidates[0]

    def _load_model(self) -> torch.nn.Module:
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        payload_config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        model_config = dict(payload_config)
        model_config.update(self.config)

        model = build_model_from_config(model_config)
        state_dict = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, Mapping) else payload
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _effective_lat_filter(self, lat_filter: tuple[float, float] | Sequence[float] | None) -> tuple[float, float] | None:
        config_filter = self.decode_config.get("lat_filter", [0.0, 40.0])
        cfg_min, cfg_max = (float(config_filter[0]), float(config_filter[1]))
        if lat_filter is None:
            lat_min, lat_max = cfg_min, cfg_max
        else:
            lat_min = max(float(lat_filter[0]), cfg_min)
            lat_max = min(float(lat_filter[1]), cfg_max)
        if lat_min > lat_max:
            return None
        return lat_min, lat_max

    def _local_msl_min(self, msl: np.ndarray, lat: float, lon: float) -> float:
        y, x = latlon_to_grid(lat, lon, self.domain, clip=True)
        cy, cx = int(round(float(y))), int(round(float(x)))

        res = max(float(self.domain.res), 1e-6)
        lat_radius_px = int(np.ceil(self.msl_min_radius_km / (111.32 * res))) + 2
        cos_lat = max(abs(float(np.cos(np.deg2rad(lat)))), 0.1)
        lon_radius_px = int(np.ceil(self.msl_min_radius_km / (111.32 * res * cos_lat))) + 2

        y0 = max(0, cy - lat_radius_px)
        y1 = min(msl.shape[0], cy + lat_radius_px + 1)
        x0 = max(0, cx - lon_radius_px)
        x1 = min(msl.shape[1], cx + lon_radius_px + 1)

        window = msl[y0:y1, x0:x1]
        if window.size == 0:
            return float("nan")

        lat_grid, lon_grid = np.meshgrid(self.lat1d[y0:y1], self.lon1d[x0:x1], indexing="ij")
        mask = haversine_km(lat_grid, lon_grid, lat, lon) <= self.msl_min_radius_km
        if not np.any(mask):
            return float(msl[cy, cx])
        return float(np.nanmin(window[mask]))

    def predict_from_grib(
        self,
        grib_file,
        patch_size: int = 40,
        eps: float = 0.1,
        lat_filter: tuple[float, float] = (0.0, 40.0),
    ) -> pd.DataFrame:
        """Predict TC centers from one AIFS GRIB2 file.

        ``patch_size`` and ``eps`` are accepted for compatibility with the
        original detector interface and are intentionally unused.
        """

        _ = (patch_size, eps)
        effective_lat_filter = self._effective_lat_filter(lat_filter)
        if effective_lat_filter is None:
            return self._empty()

        grib_path = Path(grib_file)
        valid_time, forecast_hour = parse_aifs_filename(str(grib_path))
        raw_field, _ = read_aifs_channels(
            grib_path,
            channels=self.channels,
            domain=self.domain,
            aifs_config=self.aifs_config,
        )
        field = apply_norm(raw_field, self.norm_stats)
        tensor = torch.as_tensor(field, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)

        detections = decode_heatmap(
            out["heatmap"][0, 0],
            out["offset"][0],
            self.domain,
            iso_time=valid_time.isoformat(),
            lead_hour=None,
            conf_thresh=float(self.decode_config.get("conf_thresh", 0.5)),
            lat_filter=effective_lat_filter,
            topk=self.decode_config.get("topk"),
        )
        if detections.empty:
            return self._empty()

        if "msl" in self.channels:
            msl = raw_field[self.channels.index("msl")]
            detections["MSL_MIN"] = [
                self._local_msl_min(msl, float(row.LAT), float(row.LON))
                for row in detections.itertuples(index=False)
            ]
        else:
            detections["MSL_MIN"] = np.nan

        detections["WS"] = np.nan
        return detections[self.OUTPUT_COLUMNS].reset_index(drop=True)
