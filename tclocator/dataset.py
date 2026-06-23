"""Torch datasets for ERA5/AIFS fields and synthetic smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import pandas as pd
import torch
from torch.utils.data import Dataset

from tclocator.common import DomainConfig, build_lat_lon, grid_to_latlon
from tclocator.io_era5 import read_era5_channels
from tclocator.labels import generate_labels, load_label_npz, records_at_time
from tclocator.normalization import apply_norm


@dataclass(frozen=True)
class FieldSample:
    """One input field sample."""

    path: Path
    domain_name: str
    valid_time: pd.Timestamp | None = None
    lead_hour: int | None = None
    label_path: Path | None = None


def parse_era5_valid_time_from_name(path: str | Path) -> pd.Timestamp | None:
    """Best-effort parse of YYYYMMDDHH-style ERA5 file names."""

    match = re.search(r"(\d{10}|\d{8}T\d{2}|\d{8}_\d{2}|\d{4}_\d{2}_\d{2}_\d{2})", Path(path).stem)
    if not match:
        return None
    raw = match.group(1).replace("T", "").replace("_", "")
    return pd.Timestamp(f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}T{raw[8:10]}:00:00Z")


def label_cache_path(config: Mapping[str, Any], domain_name: str, source: str | Path) -> Path:
    """Return the conventional label cache path for one source field."""

    root = Path(config.get("paths", {}).get("label_cache_dir", "data/label_cache"))
    return root / domain_name / f"{Path(source).stem}.npz"


def build_era5_samples(files: Sequence[Path], *, config: Mapping[str, Any] | None = None) -> list[FieldSample]:
    """Build ERA5 samples from files.

    ERA5 valid time is data-specific. The helper supports common
    YYYYMMDDHH-style file stems and attaches conventional cache paths when a
    config is supplied.
    """

    samples: list[FieldSample] = []
    for path in files:
        source = Path(path)
        samples.append(
            FieldSample(
                path=source,
                domain_name="era5",
                valid_time=parse_era5_valid_time_from_name(source),
                label_path=label_cache_path(config, "era5", source) if config is not None else None,
            )
        )
    return samples


def build_aifs_samples(
    files: Sequence[Path],
    *,
    lead_max: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[FieldSample]:
    """Build AIFS samples and optionally filter by forecast lead."""

    samples: list[FieldSample] = []
    for path in files:
        meta = parse_aifs_filename(path)
        if lead_max is not None and meta.forecast_hour > lead_max:
            continue
        samples.append(
            FieldSample(
                path=Path(path),
                domain_name="aifs",
                valid_time=pd.Timestamp(meta.valid_time),
                lead_hour=meta.forecast_hour,
                label_path=label_cache_path(config, "aifs", path) if config is not None else None,
            )
        )
    return samples


class FieldDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset backed by real ERA5 or AIFS field files."""

    def __init__(
        self,
        *,
        samples: Sequence[FieldSample],
        config: Mapping[str, Any],
        norm_stats: Mapping[str, Any] | None,
        ibtracs_records: pd.DataFrame | None = None,
    ) -> None:
        self.samples = list(samples)
        self.config = config
        self.domain = DomainConfig.from_mapping(config.get("domain"))
        self.channels = list(config.get("channels", []))
        self.norm_stats = norm_stats
        self.ibtracs_records = ibtracs_records
        if not self.channels:
            raise ValueError("config.channels must be non-empty")

    def __len__(self) -> int:
        """Return dataset length."""

        return len(self.samples)

    def _read_field(self, sample: FieldSample) -> tuple[np.ndarray, dict[str, Any]]:
        """Read a field sample by domain."""

        if sample.domain_name == "era5":
            return read_era5_channels(
                sample.path,
                channels=self.channels,
                domain=self.domain,
                era5_config=self.config.get("era5", {}),
            )
        if sample.domain_name == "aifs":
            return read_aifs_channels(
                sample.path,
                channels=self.channels,
                domain=self.domain,
                aifs_config=self.config.get("aifs", {}),
            )
        raise ValueError(f"Unsupported domain_name: {sample.domain_name}")

    def _make_label(self, raw_field: np.ndarray, sample: FieldSample) -> dict[str, np.ndarray]:
        """Load or generate labels for a sample."""

        if sample.label_path is not None and sample.label_path.exists():
            return load_label_npz(sample.label_path)
        if self.ibtracs_records is None or sample.valid_time is None:
            return {
                "heatmap": np.zeros(self.domain.shape, dtype=np.float32),
                "offset": np.zeros((2, self.domain.height, self.domain.width), dtype=np.float32),
                "mask": np.zeros(self.domain.shape, dtype=np.uint8),
            }
        if "msl" not in self.channels:
            raise ValueError("Label generation requires an msl channel")
        msl = raw_field[self.channels.index("msl")]
        vo = raw_field[self.channels.index("vo_850")] if "vo_850" in self.channels else None
        records = records_at_time(self.ibtracs_records, sample.valid_time)
        return generate_labels(msl=msl, vo=vo, records=records, domain=self.domain, label_config=self.config.get("labels", {}))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read, normalize, and return one training sample."""

        sample = self.samples[index]
        raw_field, meta = self._read_field(sample)
        input_field = apply_norm(raw_field, self.norm_stats) if self.norm_stats is not None else raw_field.astype(np.float32)
        label = self._make_label(raw_field, sample)
        return {
            "input": torch.from_numpy(input_field.astype(np.float32)),
            "heatmap": torch.from_numpy(label["heatmap"][None].astype(np.float32)),
            "offset": torch.from_numpy(label["offset"].astype(np.float32)),
            "mask": torch.from_numpy(label["mask"].astype(np.float32)),
            "lead_hour": torch.tensor(meta.get("forecast_hour", sample.lead_hour or -1), dtype=torch.int64),
        }


class SyntheticTCDataset(Dataset[dict[str, torch.Tensor]]):
    """Small deterministic TC-like dataset for CI and smoke tests without real data."""

    def __init__(
        self,
        *,
        length: int = 8,
        channels: Sequence[str] = ("msl", "vo_850", "t_500"),
        domain: DomainConfig | None = None,
        seed: int = 42,
        fixed_center: tuple[float, float] | None = None,
    ) -> None:
        self.length = length
        self.channels = list(channels)
        self.domain = domain or DomainConfig(lat_min=0.0, lat_max=7.75, lon_min=100.0, lon_max=115.75, res=0.25)
        self.rng = np.random.default_rng(seed)
        self.fixed_center = fixed_center
        self._samples = [self._make_sample(i) for i in range(length)]

    def __len__(self) -> int:
        """Return dataset length."""

        return self.length

    def _center_for_index(self, index: int) -> tuple[float, float]:
        """Return a synthetic center inside the domain."""

        if self.fixed_center is not None:
            return self.fixed_center
        y = 8.0 + (index * 5 % max(9, self.domain.height - 16))
        x = 10.0 + (index * 7 % max(11, self.domain.width - 20))
        return grid_to_latlon(y + 0.25, x + 0.35, self.domain)  # type: ignore[return-value]

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Create one synthetic field/label pair."""

        lat, lon = build_lat_lon(self.domain)
        yy, xx = np.indices(self.domain.shape, dtype=np.float32)
        center_lat, center_lon = self._center_for_index(index)
        cy = (self.domain.lat_max - center_lat) / self.domain.res
        cx = (center_lon - self.domain.lon_min) / self.domain.res
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        vortex = np.exp(-r2 / (2.0 * 3.0**2)).astype(np.float32)

        raw_by_channel = {
            "msl": (101000.0 - 3500.0 * vortex + 15.0 * yy).astype(np.float32),
            "vo_850": (1.2e-4 * vortex).astype(np.float32),
            "t_500": (260.0 + 4.0 * vortex - 0.03 * yy).astype(np.float32),
        }
        field = np.stack([raw_by_channel.get(channel, vortex) for channel in self.channels], axis=0).astype(np.float32)
        record = [{"SID": f"SYN{index:03d}", "LAT": center_lat, "LON": center_lon, "ISO_TIME": datetime(2000, 1, 1)}]
        label = generate_labels(
            msl=raw_by_channel["msl"],
            records=record,
            domain=self.domain,
            label_config={"mode": "ibtracs", "sigma_px": 2.0, "search_radius_km": 300.0},
        )
        return {
            "input": torch.from_numpy(field),
            "heatmap": torch.from_numpy(label["heatmap"][None]),
            "offset": torch.from_numpy(label["offset"]),
            "mask": torch.from_numpy(label["mask"].astype(np.float32)),
            "lead_hour": torch.tensor(0, dtype=torch.int64),
            "center": torch.tensor([center_lat, center_lon], dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one synthetic sample."""

        return self._samples[index]


def collate_batch(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dictionary tensors for DataLoader."""

    keys = items[0].keys()
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in items if isinstance(item.get(key), torch.Tensor)]
        if values:
            batch[key] = torch.stack(values, dim=0)
    return batch
