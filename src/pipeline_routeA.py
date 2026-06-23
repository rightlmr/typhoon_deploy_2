"""Route A deployment pipeline: heatmap localization followed by intensity correction."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
import yaml

from aifs_data_utils import (
    find_aifs_grib_by_init,
    find_files_by_init_date,
    find_latest_init_time_files,
    parse_aifs_filename,
    scan_aifs_grib_files,
)
from detector_adapter import HeatmapDetectorAdapter


def _resolve_path(root: Path, value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ISO_TIME",
        "LAT",
        "LON",
        "MSL_MIN",
        "CONF",
        "PRED_WIND",
        "PRED_PRES",
        "GRIB_FILE",
        "SOURCE_FILE",
        "INIT_TIME",
        "FORECAST_HOUR",
    ])


def build_history_for_detection(
    detection: pd.Series,
    history_timesteps: List[int],
    aifs_data_dir: str,
    init_time: datetime,
    forecast_hour: int,
) -> Optional[List[Dict[str, Any]]]:
    """Build same-init or previous-init history fields for one detection."""

    history_info: list[dict[str, Any]] = []
    for dt in history_timesteps:
        target_fh = forecast_hour - dt
        if target_fh >= 0:
            target_init = init_time
        else:
            target_init = init_time - timedelta(days=1)
            target_fh = 24 + target_fh

        grib_path = find_aifs_grib_by_init(aifs_data_dir, target_init, target_fh)
        if grib_path is None:
            return None

        history_info.append({
            "grib_path": grib_path,
            "forecast_hour": target_fh,
            "lat": detection["LAT"],
            "lon": detection["LON"],
        })
    return history_info


class TyphoonRouteAPipeline:
    """Heatmap detector + per-point correction, without V9 tracking/filtering."""

    def __init__(
        self,
        detection_model_dir: str,
        detection_config: Mapping[str, Any],
        correction_ckpt: str | None,
        correction_config: str | None,
        aifs_stats_file: str | None,
        ibtracs_stats_file: str | None,
        aifs_grib_dir: str,
        correction_enabled: bool = True,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.aifs_data_dir = aifs_grib_dir

        print("=" * 60)
        print("Initializing Typhoon Route A Pipeline")
        print("=" * 60)

        print("\n[1/2] Loading heatmap detection model...")
        self.detection_infer = HeatmapDetectorAdapter(detection_model_dir, device=device, config=detection_config)

        self.model_args: dict[str, Any] = {}
        self.data_args: dict[str, Any] = {}
        self.correction_infer = None
        self.use_history = False
        self.history_timesteps: list[int] = []
        self.field_size = None
        self.correction_enabled = bool(correction_enabled)

        if self.correction_enabled:
            missing = [
                p for p in [correction_ckpt, correction_config, aifs_stats_file, ibtracs_stats_file]
                if not p or not Path(p).exists()
            ]
            if missing:
                print("[WARN] Correction assets missing; Route A will output localization only")
                for item in missing:
                    print(f"       missing: {item}")
                self.correction_enabled = False

        if self.correction_enabled:
            print("\n[2/2] Loading intensity correction model...")
            from tc_correction_model import TyphoonCorrectionInference

            with open(correction_config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            self.model_args = config.get("model_args", {})
            self.data_args = config.get("data_args", {})
            self.correction_infer = TyphoonCorrectionInference(
                checkpoint_path=correction_ckpt,
                model_args=self.model_args,
                data_args=self.data_args,
                aifs_stats_file=aifs_stats_file,
                ibtracs_stats_file=ibtracs_stats_file,
                device=device,
            )
            self.use_history = self.correction_infer.use_history
            self.history_timesteps = self.correction_infer.history_timesteps
            self.field_size = self.correction_infer.field_size
        else:
            print("\n[2/2] Correction disabled")

        print("\n" + "=" * 60)
        print("Pipeline initialization complete")
        print(f"  Device: {self.device}")
        print(f"  Correction enabled: {self.correction_enabled}")
        print(f"  Use history: {self.use_history}")
        if self.use_history:
            print(f"  History timesteps: {self.history_timesteps}")
        print("=" * 60)

    def _attach_detection_metadata(self, detections: pd.DataFrame, grib_file: str) -> pd.DataFrame:
        detections = detections.copy()
        valid_time, forecast_hour = parse_aifs_filename(grib_file)
        init_time = valid_time - timedelta(hours=forecast_hour)
        detections["GRIB_FILE"] = grib_file
        detections["SOURCE_FILE"] = os.path.basename(grib_file)
        detections["INIT_TIME"] = init_time
        detections["FORECAST_HOUR"] = forecast_hour
        return detections

    def _run_correction_for_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        if detections is None or len(detections) == 0:
            return _empty_results()

        if not self.correction_enabled:
            out = detections.copy()
            out["PRED_WIND"] = np.nan
            out["PRED_PRES"] = np.nan
            return out

        from tc_correction_model import build_correction_input

        results = []
        for idx, det in detections.iterrows():
            grib_file = det.get("GRIB_FILE", None)
            if not grib_file or not os.path.exists(grib_file):
                print(f"[WARN] Missing GRIB_FILE for detection {idx}, skipping correction")
                continue

            valid_time, forecast_hour = parse_aifs_filename(grib_file)
            init_time = valid_time - timedelta(hours=forecast_hour)
            pred_wind, pred_pres = np.nan, np.nan

            try:
                history_info = None
                if self.use_history:
                    history_info = build_history_for_detection(
                        det,
                        self.history_timesteps,
                        self.aifs_data_dir,
                        init_time,
                        forecast_hour,
                    )
                    if history_info is None:
                        print(f"[WARN] History missing for detection {idx}, correction skipped")

                if (not self.use_history) or history_info is not None:
                    inputs = build_correction_input(
                        grib_path=grib_file,
                        forecast_hour=forecast_hour,
                        lat=float(det["LAT"]),
                        lon=float(det["LON"]),
                        aifs_stats=self.correction_infer.aifs_stats,
                        ibtracs_stats=self.correction_infer.ibtracs_stats,
                        var_order=self.correction_infer.var_order,
                        field_size=self.field_size,
                        history_info=history_info,
                    )
                    pred = self.correction_infer.predict_and_denormalize(inputs)
                    pred_wind = float(pred[0, 0])
                    pred_pres = float(pred[0, 1])
            except Exception as exc:
                print(f"[WARN] Correction failed for detection {idx}: {exc}")

            row = det.to_dict()
            row.update({"PRED_WIND": pred_wind, "PRED_PRES": pred_pres})
            results.append(row)

        return pd.DataFrame(results)

    def run_on_file(
        self,
        grib_file: str,
        patch_size: int = 40,
        eps: float = 0.1,
        lat_filter: Tuple[float, float] = (0.0, 40.0),
    ) -> pd.DataFrame:
        print(f"\n[Pipeline] Processing: {os.path.basename(grib_file)}")
        detections = self.detection_infer.predict_from_grib(
            grib_file,
            patch_size=patch_size,
            eps=eps,
            lat_filter=lat_filter,
        )
        if detections.empty:
            print("[Pipeline] No detections found")
            return _empty_results()

        detections = self._attach_detection_metadata(detections, grib_file)
        result_df = self._run_correction_for_detections(detections)
        print(f"[Pipeline] Results: {len(result_df)} rows")
        return result_df

    def _select_grib_files(self, aifs_dir: str, init_date) -> list[str]:
        if init_date is None:
            grib_files = scan_aifs_grib_files(aifs_dir)
            print(f"[Pipeline] Found {len(grib_files)} GRIB2 files in {aifs_dir}")
            return grib_files

        if isinstance(init_date, str) and init_date.lower() == "latest":
            latest_init, grib_files = find_latest_init_time_files(aifs_dir)
            if latest_init is None:
                print(f"[Pipeline] No GRIB2 files found in {aifs_dir}")
                return []
            print(f"[Pipeline] Latest init time: {latest_init:%Y-%m-%d %H:%M}, {len(grib_files)} files selected")
            return grib_files

        if isinstance(init_date, str):
            init_date_str = init_date.strip()
            try:
                if " " in init_date_str or "T" in init_date_str:
                    init_date = datetime.strptime(init_date_str.replace("T", " "), "%Y-%m-%d %H")
                else:
                    init_date = datetime.strptime(init_date_str, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError(
                    f"Invalid init_date format: {init_date!r}. "
                    "Supported: latest, YYYY-MM-DD, YYYY-MM-DD HH"
                ) from exc

        matched_init, grib_files = find_files_by_init_date(aifs_dir, init_date)
        if matched_init is None:
            print(f"[Pipeline] No GRIB2 files found for init_date={init_date}")
            return []
        print(f"[Pipeline] Matched init time: {matched_init:%Y-%m-%d %H:%M}, {len(grib_files)} files selected")
        return grib_files

    def run_on_directory(
        self,
        aifs_dir: str,
        output_dir: str,
        patch_size: int = 40,
        eps: float = 0.1,
        lat_filter: Tuple[float, float] = (0.0, 40.0),
        lead_min: Optional[int] = None,
        lead_max: Optional[int] = None,
        init_date=None,
    ) -> pd.DataFrame:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        grib_files = self._select_grib_files(aifs_dir, init_date)

        if lead_min is not None or lead_max is not None:
            filtered = []
            for path in grib_files:
                try:
                    _, lead = parse_aifs_filename(path)
                except ValueError:
                    continue
                if (lead_min is None or lead >= lead_min) and (lead_max is None or lead <= lead_max):
                    filtered.append(path)
            grib_files = filtered
            print(f"[Pipeline] After lead filter: {len(grib_files)} files")

        raw_detections = []
        for grib_file in grib_files:
            try:
                print(f"\n[Pipeline] Detecting: {os.path.basename(grib_file)}")
                detections = self.detection_infer.predict_from_grib(
                    grib_file,
                    patch_size=patch_size,
                    eps=eps,
                    lat_filter=lat_filter,
                )
                if not detections.empty:
                    raw_detections.append(self._attach_detection_metadata(detections, grib_file))
            except Exception as exc:
                print(f"[ERROR] Failed to process {grib_file}: {exc}")
                import traceback

                traceback.print_exc()

        if not raw_detections:
            print("\n[Pipeline] No raw detections generated")
            empty = _empty_results()
            empty.to_csv(Path(output_dir) / "all_results.csv", index=False)
            return empty

        raw = pd.concat(raw_detections, ignore_index=True)
        raw_path = Path(output_dir) / "detections_raw.csv"
        raw.to_csv(raw_path, index=False)
        print(f"\n[Pipeline] Raw detections saved: {raw_path}")
        print(f"[Pipeline] Raw detections: {len(raw)} rows, {raw['ISO_TIME'].nunique()} times")

        if self.correction_enabled:
            print("\n[Pipeline] Running intensity correction on raw Route A detections...")
        else:
            print("\n[Pipeline] Correction disabled; saving localization rows with NaN intensity...")
        combined = self._run_correction_for_detections(raw)

        combined_path = Path(output_dir) / "all_results.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[Pipeline] All results saved: {combined_path}")
        print(f"[Pipeline] Total: {len(combined)} Route A detections")
        return combined


def load_config_from_yaml(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Typhoon Route A heatmap detection + correction pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--aifs_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--init_date", default=None)
    parser.add_argument("--lead_min", type=int, default=None)
    parser.add_argument("--lead_max", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    root = config_path.parent
    config = load_config_from_yaml(config_path)
    det_cfg = dict(config.get("detection_model", {}))
    corr_cfg = dict(config.get("correction_model", {}))
    stats_cfg = dict(config.get("stats", {}))
    data_cfg = dict(config.get("data", {}))
    output_cfg = dict(config.get("output", {}))
    inf_cfg = dict(config.get("inference", {}))

    aifs_dir = _resolve_path(root, args.aifs_dir or data_cfg.get("aifs_grib_dir"))
    output_dir = _resolve_path(root, args.output_dir or output_cfg.get("output_dir", "output"))
    detection_model_dir = _resolve_path(root, det_cfg.get("model_dir", "checkpoints/detection"))
    correction_ckpt = _resolve_path(root, corr_cfg.get("checkpoint"))
    correction_config = _resolve_path(root, corr_cfg.get("config"))
    aifs_stats_file = _resolve_path(root, stats_cfg.get("aifs_stats"))
    ibtracs_stats_file = _resolve_path(root, stats_cfg.get("ibtracs_stats"))

    if not aifs_dir:
        parser.error("--aifs_dir is required, or set data.aifs_grib_dir in config")
    if not detection_model_dir:
        parser.error("detection_model.model_dir is required")

    decode_cfg = dict(det_cfg.get("decode", {}))
    cfg_lat_filter = decode_cfg.get("lat_filter", [0.0, 40.0])
    lat_filter = (
        float(args.lat_min if args.lat_min is not None else cfg_lat_filter[0]),
        float(args.lat_max if args.lat_max is not None else cfg_lat_filter[1]),
    )

    pipeline = TyphoonRouteAPipeline(
        detection_model_dir=detection_model_dir,
        detection_config=det_cfg,
        correction_ckpt=correction_ckpt,
        correction_config=correction_config,
        aifs_stats_file=aifs_stats_file,
        ibtracs_stats_file=ibtracs_stats_file,
        aifs_grib_dir=aifs_dir,
        correction_enabled=bool(corr_cfg.get("enabled", True)),
        device=args.device or inf_cfg.get("device", "auto"),
    )

    init_date = args.init_date if args.init_date is not None else data_cfg.get("init_date")
    results = pipeline.run_on_directory(
        aifs_dir=aifs_dir,
        output_dir=output_dir,
        patch_size=int(args.patch_size if args.patch_size is not None else det_cfg.get("patch_size", 40)),
        eps=float(args.eps if args.eps is not None else det_cfg.get("eps", 0.1)),
        lat_filter=lat_filter,
        lead_min=args.lead_min if args.lead_min is not None else inf_cfg.get("lead_min"),
        lead_max=args.lead_max if args.lead_max is not None else inf_cfg.get("lead_max"),
        init_date=init_date,
    )

    print("\n" + "=" * 60)
    print("Inference Complete")
    print("=" * 60)
    if len(results) > 0:
        print(results.head(10).to_string())
        print(f"\nTotal detections: {len(results)}")
        print(f"Valid wind predictions: {results['PRED_WIND'].notna().sum()}")
        print(f"Valid pres predictions: {results['PRED_PRES'].notna().sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
