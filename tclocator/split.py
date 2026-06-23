"""Group-aware train/validation splitting utilities."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import pandas as pd

from tclocator.dataset import FieldSample, build_aifs_samples
from tclocator.io_aifs import parse_aifs_filename


def aifs_init_month(path: str | Path) -> str:
    """Return the AIFS initialization month as ``YYYY-MM``."""

    meta = parse_aifs_filename(path)
    return meta.init_time.strftime("%Y-%m")


def usable_months_from_config(config: Mapping[str, Any]) -> set[str] | None:
    """Return explicitly configured usable AIFS months, or ``None`` if unset."""

    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        return None
    raw = data_cfg.get("usable_months")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    months = {str(month).replace("_", "-") for month in raw}
    return months


def filter_aifs_files_by_usable_months(config: Mapping[str, Any], files: Sequence[Path]) -> list[Path]:
    """Filter AIFS files to configured usable initialization months."""

    months = usable_months_from_config(config)
    paths = [Path(path) for path in files]
    if months is None:
        return paths
    kept: list[Path] = []
    for path in paths:
        try:
            month = aifs_init_month(path)
        except ValueError:
            continue
        if month in months:
            kept.append(path)
    return kept


def sample_group_id(sample: FieldSample, group_by: str) -> str | None:
    """Return the leakage group identifier for one sample."""

    valid_time = sample.valid_time
    if valid_time is None:
        return None
    timestamp = pd.Timestamp(valid_time)
    if group_by == "init_time":
        lead = int(sample.lead_hour or 0)
        return (timestamp - timedelta(hours=lead)).isoformat()
    if group_by == "valid_date":
        return timestamp.strftime("%Y-%m-%d")
    if group_by == "year_month":
        lead = int(sample.lead_hour or 0)
        return (timestamp - timedelta(hours=lead)).strftime("%Y-%m")
    raise ValueError(f"Unsupported split.group_by: {group_by}")


def grouped_split(
    samples: Sequence[FieldSample],
    *,
    group_by: str,
    val_fraction: float,
    seed: int,
    val_groups_override: Sequence[str] | None = None,
) -> tuple[list[int], list[int], list[str]]:
    """Split sample indices by whole groups, with ungroupable samples in train."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("split.val_fraction must satisfy 0 <= val_fraction < 1")
    group_ids = [sample_group_id(sample, group_by) for sample in samples]
    unique_groups = sorted({group for group in group_ids if group is not None})
    if val_groups_override:
        val_groups = {str(group) for group in val_groups_override}
    else:
        rng = random.Random(seed)
        rng.shuffle(unique_groups)
        n_val = max(1, round(len(unique_groups) * val_fraction)) if unique_groups and val_fraction > 0.0 else 0
        val_groups = set(unique_groups[:n_val])
    train_idx = [idx for idx, group in enumerate(group_ids) if group not in val_groups]
    val_idx = [idx for idx, group in enumerate(group_ids) if group is not None and group in val_groups]
    return train_idx, val_idx, sorted(val_groups)


def split_config(config: Mapping[str, object], default_group_by: str) -> tuple[str, float, int]:
    """Return normalized split settings from a project config."""

    split = config.get("split", {})
    if not isinstance(split, Mapping):
        split = {}
    group_by = str(split.get("group_by", default_group_by))
    val_fraction = float(split.get("val_fraction", 0.2))
    seed = int(split.get("seed", config.get("seed", 42)))
    return group_by, val_fraction, seed


def split_val_groups_override(config: Mapping[str, object]) -> list[str] | None:
    """Return optional validation group override from config."""

    split = config.get("split", {})
    if not isinstance(split, Mapping):
        return None
    raw = split.get("val_groups_override")
    if raw is None:
        return None
    return [str(group) for group in raw]


def select_aifs_files(config: Mapping[str, object], files: Sequence[Path], which: str) -> list[Path]:
    """Select AIFS files from the deterministic train/validation split."""

    if which not in {"all", "train", "val"}:
        raise ValueError(f"Unsupported split selection: {which}")
    paths = filter_aifs_files_by_usable_months(config, [Path(path) for path in files])
    if which == "all":
        return paths
    group_by, val_fraction, seed = split_config(config, "init_time")
    samples = build_aifs_samples(paths, lead_max=None, config=config)
    train_idx, val_idx, _ = grouped_split(
        samples,
        group_by=group_by,
        val_fraction=val_fraction,
        seed=seed,
        val_groups_override=split_val_groups_override(config),
    )
    selected = train_idx if which == "train" else val_idx
    return [samples[idx].path for idx in selected]
