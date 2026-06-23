"""Per-domain channel normalization statistics and transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def channel_method(channel: str, norm_config: Mapping[str, Any]) -> str:
    """Resolve the normalization method for a channel."""

    channel_methods = norm_config.get("channel_methods", {})
    if isinstance(channel_methods, Mapping) and channel in channel_methods:
        return str(channel_methods[channel])
    return str(norm_config.get("method", "zscore"))


def _first_pass_shift(samples: Sequence[np.ndarray], channels: Sequence[str], norm_config: Mapping[str, Any]) -> dict[str, float]:
    """Calculate non-negative shifts for log1p normalization."""

    shifts: dict[str, float] = {}
    for ci, channel in enumerate(channels):
        method = channel_method(channel, norm_config)
        if method == "log1p+zscore":
            min_value = min(float(np.nanmin(sample[ci])) for sample in samples)
            shifts[channel] = max(0.0, -min_value)
        else:
            shifts[channel] = 0.0
    return shifts


def _transform(values: np.ndarray, method: str, shift: float) -> np.ndarray:
    """Apply the pre-zscore transformation for a channel."""

    arr = values.astype(np.float64, copy=False)
    if method == "zscore":
        return arr
    if method == "log1p+zscore":
        return np.log1p(np.maximum(arr + shift, 0.0))
    raise ValueError(f"Unsupported normalization method: {method}")


def compute_norm_stats(
    samples: Sequence[np.ndarray],
    channels: Sequence[str],
    norm_config: Mapping[str, Any],
    *,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute per-channel mean/std stats from ``[C,H,W]`` samples."""

    if not samples:
        raise ValueError("Cannot compute normalization stats from zero samples")
    shifts = _first_pass_shift(samples, channels, norm_config)
    stats: dict[str, Any] = {"channels": list(channels), "stats": {}}

    for ci, channel in enumerate(channels):
        method = channel_method(channel, norm_config)
        shift = shifts[channel]
        total = 0
        sum_value = 0.0
        sum_sq = 0.0
        for sample in samples:
            values = _transform(sample[ci], method, shift)
            total += values.size
            sum_value += float(np.nansum(values))
            sum_sq += float(np.nansum(values * values))
        mean = sum_value / max(total, 1)
        var = max(sum_sq / max(total, 1) - mean * mean, 0.0)
        std = max(float(np.sqrt(var)), eps)
        stats["stats"][channel] = {"method": method, "mean": mean, "std": std, "shift": shift}
    return stats


def compute_norm_stats_stream(
    sample_iter_factory: Callable[[], Iterable[np.ndarray]],
    channels: Sequence[str],
    norm_config: Mapping[str, Any],
    *,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute per-channel stats from a re-iterable sample source.

    This avoids materializing thousands of meteorological fields in memory.
    The factory is called once for log-shift discovery and once for moments.
    """

    methods = {channel: channel_method(channel, norm_config) for channel in channels}
    shifts = {channel: 0.0 for channel in channels}
    if any(method == "log1p+zscore" for method in methods.values()):
        seen = False
        mins = {channel: np.inf for channel in channels}
        for sample in sample_iter_factory():
            seen = True
            for ci, channel in enumerate(channels):
                if methods[channel] == "log1p+zscore":
                    mins[channel] = min(mins[channel], float(np.nanmin(sample[ci])))
        if not seen:
            raise ValueError("Cannot compute normalization stats from zero samples")
        shifts = {channel: max(0.0, -mins[channel]) if methods[channel] == "log1p+zscore" else 0.0 for channel in channels}

    totals = {channel: 0 for channel in channels}
    sums = {channel: 0.0 for channel in channels}
    sum_sqs = {channel: 0.0 for channel in channels}
    seen = False
    for sample in sample_iter_factory():
        seen = True
        for ci, channel in enumerate(channels):
            values = _transform(sample[ci], methods[channel], shifts[channel])
            totals[channel] += values.size
            sums[channel] += float(np.nansum(values))
            sum_sqs[channel] += float(np.nansum(values * values))
    if not seen:
        raise ValueError("Cannot compute normalization stats from zero samples")

    stats: dict[str, Any] = {"channels": list(channels), "stats": {}}
    for channel in channels:
        mean = sums[channel] / max(totals[channel], 1)
        var = max(sum_sqs[channel] / max(totals[channel], 1) - mean * mean, 0.0)
        stats["stats"][channel] = {
            "method": methods[channel],
            "mean": mean,
            "std": max(float(np.sqrt(var)), eps),
            "shift": shifts[channel],
        }
    return stats


def apply_norm(sample: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Apply stored per-channel normalization to a ``[C,H,W]`` sample."""

    channels = list(stats["channels"])
    if sample.shape[0] != len(channels):
        raise ValueError(f"Sample has {sample.shape[0]} channels, stats has {len(channels)}")
    out = np.empty_like(sample, dtype=np.float32)
    for ci, channel in enumerate(channels):
        spec = stats["stats"][channel]
        values = _transform(sample[ci], str(spec["method"]), float(spec.get("shift", 0.0)))
        out[ci] = ((values - float(spec["mean"])) / float(spec["std"])).astype(np.float32)
    return out


def save_norm_stats(stats: Mapping[str, Any], path: str | Path) -> None:
    """Save normalization stats as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    """Load normalization stats from JSON."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_samples(reader: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Materialize reader output as float32 samples for stats computation."""

    return [np.asarray(sample, dtype=np.float32) for sample in reader]
