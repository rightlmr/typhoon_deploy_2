"""Greedy nearest-neighbor track association for decoded detections."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tclocator.common import haversine_km


@dataclass
class _TrackState:
    """Internal active track state."""

    track_id: int
    rows: list[int] = field(default_factory=list)
    last_time: pd.Timestamp | None = None
    last_lat: float = 0.0
    last_lon: float = 0.0


def link_tracks(
    detections: pd.DataFrame,
    *,
    max_step_km: float = 800.0,
    min_len: int = 4,
    expected_step_hours: float = 6.0,
) -> pd.DataFrame:
    """Associate detections across time using greedy nearest-neighbor links."""

    if detections.empty:
        out = detections.copy()
        out["TRACK_ID"] = pd.Series(dtype="int64")
        return out

    df = detections.copy().reset_index(drop=True)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df = df.sort_values(["ISO_TIME", "CONF"], ascending=[True, False]).reset_index(drop=True)
    df["TRACK_ID"] = -1
    active: list[_TrackState] = []
    next_track_id = 0

    for time, group in df.groupby("ISO_TIME", sort=True):
        used_tracks: set[int] = set()
        for idx, row in group.sort_values("CONF", ascending=False).iterrows():
            best_track: _TrackState | None = None
            best_dist = float("inf")
            for track in active:
                if track.track_id in used_tracks or track.last_time is None:
                    continue
                hours = max((time - track.last_time).total_seconds() / 3600.0, expected_step_hours)
                allowed = max_step_km * (hours / expected_step_hours)
                dist = haversine_km(track.last_lat, track.last_lon, float(row["LAT"]), float(row["LON"]))
                if dist <= allowed and dist < best_dist:
                    best_dist = float(dist)
                    best_track = track
            if best_track is None:
                best_track = _TrackState(track_id=next_track_id)
                next_track_id += 1
                active.append(best_track)
            best_track.rows.append(int(idx))
            best_track.last_time = time
            best_track.last_lat = float(row["LAT"])
            best_track.last_lon = float(row["LON"])
            used_tracks.add(best_track.track_id)
            df.loc[idx, "TRACK_ID"] = best_track.track_id

        active = [
            track
            for track in active
            if track.last_time is not None and (time - track.last_time).total_seconds() <= expected_step_hours * 3600.0 * 4.0
        ]

    counts = df["TRACK_ID"].value_counts()
    keep_ids = set(counts[counts >= min_len].index.tolist())
    return df.loc[df["TRACK_ID"].isin(keep_ids)].reset_index(drop=True)

