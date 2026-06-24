"""One-command integration gate checks for the Route A deployment package."""

from __future__ import annotations

import warnings

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.*")
        import pygrib  # noqa: F401
except ImportError:
    pygrib = None  # type: ignore[assignment]

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detector_adapter import HeatmapDetectorAdapter
from pipeline_routeA import TyphoonRouteAPipeline, load_config_from_yaml
from tclocator.common import DomainConfig, build_lat_lon, haversine_km
from tclocator.io_aifs import read_aifs_channels


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIPPED = "SKIPPED"


@dataclass
class GateResult:
    gate: str
    status: str
    metric: str
    hint: str = ""


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def print_result(result: GateResult) -> None:
    print(f"[{result.status}] {result.gate}: {result.metric}")
    if result.hint:
        print(f"       {result.hint}")


def run_gate(results: list[GateResult], name: str, fn: Callable[[], GateResult]) -> GateResult:
    try:
        result = fn()
    except Exception as exc:
        print(f"[ERROR] {name} raised: {exc}")
        traceback.print_exc(limit=5)
        result = GateResult(name, FAIL, str(exc))
    results.append(result)
    print_result(result)
    return result


def cfg_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def build_pipeline_kwargs(root: Path, config: Mapping[str, Any], device: str | None = None) -> dict[str, Any]:
    det_cfg = dict(config.get("detection_model", {}))
    corr_cfg = dict(config.get("correction_model", {}))
    stats_cfg = dict(config.get("stats", {}))
    data_cfg = dict(config.get("data", {}))
    inf_cfg = dict(config.get("inference", {}))

    aifs_dir = resolve_path(root, data_cfg.get("aifs_grib_dir"))
    return {
        "detection_model_dir": str(resolve_path(root, det_cfg.get("model_dir", "checkpoints/detection"))),
        "detection_config": det_cfg,
        "correction_ckpt": str(resolve_path(root, corr_cfg.get("checkpoint"))) if corr_cfg.get("checkpoint") else None,
        "correction_config": str(resolve_path(root, corr_cfg.get("config"))) if corr_cfg.get("config") else None,
        "aifs_stats_file": str(resolve_path(root, stats_cfg.get("aifs_stats"))) if stats_cfg.get("aifs_stats") else None,
        "ibtracs_stats_file": str(resolve_path(root, stats_cfg.get("ibtracs_stats"))) if stats_cfg.get("ibtracs_stats") else None,
        "aifs_grib_dir": str(aifs_dir) if aifs_dir else "",
        "correction_enabled": bool(corr_cfg.get("enabled", True)),
        "device": device or inf_cfg.get("device", "auto"),
    }


def top_detection_metric(detections) -> str:
    if detections is None or detections.empty:
        return "rows=0"
    top = detections.sort_values("CONF", ascending=False).iloc[0]
    return (
        f"rows={len(detections)}, top=(lat={float(top['LAT']):.3f}, "
        f"lon={float(top['LON']):.3f}, conf={float(top.get('CONF', np.nan)):.4f}, "
        f"msl_min={float(top.get('MSL_MIN', np.nan)):.1f})"
    )


def diagnose_msl_alignment(grib_path: Path, config: Mapping[str, Any], truth_lat: float, truth_lon: float) -> str:
    det_cfg = dict(config.get("detection_model", {}))
    domain = DomainConfig.from_mapping(det_cfg.get("domain"))
    channels = list(det_cfg.get("channels", ["msl", "vo_850", "t_500"]))
    if "msl" not in channels:
        return "MSL channel not configured; cannot diagnose pressure-field alignment."

    field, _ = read_aifs_channels(
        grib_path,
        channels=channels,
        domain=domain,
        aifs_config=det_cfg.get("aifs", {}),
    )
    msl = field[channels.index("msl")]
    lat1d, lon1d = build_lat_lon(domain)
    min_y, min_x = np.unravel_index(np.nanargmin(msl), msl.shape)
    min_lat = float(lat1d[min_y])
    min_lon = float(lon1d[min_x])
    min_dist = haversine_km(min_lat, min_lon, truth_lat, truth_lon)

    truth_y = int(round((domain.lat_max - truth_lat) / domain.res))
    truth_x = int(round((truth_lon % 360.0 - domain.lon_min) / domain.res))
    truth_y = max(0, min(msl.shape[0] - 1, truth_y))
    truth_x = max(0, min(msl.shape[1] - 1, truth_x))
    truth_msl_hpa = float(msl[truth_y, truth_x]) / 100.0

    return (
        "diagnostic: "
        f"msl_global_min=(lat={min_lat:.3f}, lon={min_lon:.3f}, "
        f"dist_to_truth={min_dist:.1f}km, value={float(msl[min_y, min_x]) / 100.0:.1f}hPa); "
        f"msl_at_truth={truth_msl_hpa:.1f}hPa. "
        "If these are hundreds of km away from truth, inspect GRIB latitude order and longitude wrapping."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run typhoon_deploy_2 integration gates on a Linux target.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--grib", required=True, help="Known strong-storm short-lead AIFS GRIB2 file")
    parser.add_argument("--truth-lat", type=float, required=True)
    parser.add_argument("--truth-lon", type=float, required=True)
    parser.add_argument("--truth-name", default="")
    parser.add_argument("--dir", default=None, help="Optional GRIB directory for full latest-init directory gate")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    root = config_path.parent
    config = load_config_from_yaml(config_path)
    grib_path = resolve_path(root, args.grib)
    if grib_path is None:
        raise ValueError("--grib is required")
    truth_lat = float(args.truth_lat)
    truth_lon = float(args.truth_lon) % 360.0

    print("=" * 72)
    print("typhoon_deploy_2 integration gate check")
    print("=" * 72)
    print(f"project_root={root}")
    print(f"config={config_path}")
    print(f"grib={grib_path}")
    print(f"truth={args.truth_name or 'unnamed'} lat={truth_lat:.3f}, lon={truth_lon:.3f}")
    print()

    results: list[GateResult] = []
    state: dict[str, Any] = {}

    def gate0() -> GateResult:
        cuda_ok = torch.cuda.is_available()
        pygrib_ok = pygrib is not None
        metric = f"torch={torch.__version__}, cuda_available={cuda_ok}, pygrib={pygrib_ok}"
        if not pygrib_ok:
            return GateResult("Gate 0 environment", FAIL, metric, "Install libeccodes and pygrib before GRIB gates.")
        return GateResult("Gate 0 environment", PASS, metric)

    gate0_result = run_gate(results, "Gate 0 environment", gate0)

    def gate1() -> GateResult:
        det_ckpt = resolve_path(root, cfg_get(config, "detection_model", "checkpoint", default=None))
        if det_ckpt is None:
            det_ckpt = resolve_path(root, Path(cfg_get(config, "detection_model", "model_dir", default="checkpoints/detection")) / "finetune_best.ckpt")
        corr_ckpt = resolve_path(root, cfg_get(config, "correction_model", "checkpoint", default=None))

        loaded = []
        for label, path in (("detection", det_ckpt), ("correction", corr_ckpt)):
            if path is None or not path.exists():
                return GateResult("Gate 1 checkpoint load", FAIL, f"{label} missing: {path}")
            payload = torch.load(path, map_location="cpu")
            keys = sorted(payload.keys())[:6] if isinstance(payload, Mapping) else [type(payload).__name__]
            loaded.append(f"{label}={path.name} keys={keys}")
        return GateResult("Gate 1 checkpoint load", PASS, "; ".join(loaded))

    run_gate(results, "Gate 1 checkpoint load", gate1)

    def gate2() -> GateResult:
        kwargs = build_pipeline_kwargs(root, config, args.device)
        pipeline = TyphoonRouteAPipeline(**kwargs)
        state["pipeline"] = pipeline
        if not pipeline.correction_enabled:
            return GateResult("Gate 2 dependency coexistence", FAIL, "pipeline initialized but correction_enabled=False")
        return GateResult(
            "Gate 2 dependency coexistence",
            PASS,
            f"device={pipeline.device}, use_history={pipeline.use_history}, history={pipeline.history_timesteps}",
        )

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 2 dependency coexistence", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 2 dependency coexistence", gate2)

    def gate3() -> GateResult:
        if not grib_path.exists():
            return GateResult("Gate 3 detector interface", FAIL, f"GRIB not found: {grib_path}")
        pipeline = state.get("pipeline")
        if pipeline is not None:
            adapter = pipeline.detection_infer
        else:
            det_cfg = dict(config.get("detection_model", {}))
            adapter = HeatmapDetectorAdapter(
                str(resolve_path(root, det_cfg.get("model_dir", "checkpoints/detection"))),
                device=args.device or cfg_get(config, "inference", "device", default="auto"),
                config=det_cfg,
            )
        lat_filter = tuple(cfg_get(config, "detection_model", "decode", "lat_filter", default=[0.0, 40.0]))
        detections = adapter.predict_from_grib(grib_path, lat_filter=lat_filter)
        state["detections"] = detections
        required = {"ISO_TIME", "LAT", "LON", "MSL_MIN", "WS"}
        missing = sorted(required - set(detections.columns))
        if missing:
            return GateResult("Gate 3 detector interface", FAIL, f"missing columns={missing}")
        status = WARN if detections.empty else PASS
        hint = "No detections; use a stronger/shorter-lead GRIB case for Gate 4." if detections.empty else ""
        return GateResult("Gate 3 detector interface", status, top_detection_metric(detections), hint)

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 3 detector interface", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 3 detector interface", gate3)

    def gate4() -> GateResult:
        detections = state.get("detections")
        if detections is None or detections.empty:
            return GateResult("Gate 4 coordinate alignment", SKIPPED, "no Gate 3 top detection")
        top = detections.sort_values("CONF", ascending=False).iloc[0]
        pred_lat = float(top["LAT"])
        pred_lon = float(top["LON"]) % 360.0
        dist = haversine_km(pred_lat, pred_lon, truth_lat, truth_lon)
        metric = (
            f"truth=({truth_lat:.3f},{truth_lon:.3f}), "
            f"pred_top=({pred_lat:.3f},{pred_lon:.3f}), dist={dist:.1f}km"
        )
        if dist < 80.0:
            return GateResult("Gate 4 coordinate alignment", PASS, metric)
        if dist < 300.0:
            return GateResult("Gate 4 coordinate alignment", WARN, metric, "Borderline case; repeat with another strong short-lead sample.")
        return GateResult("Gate 4 coordinate alignment", FAIL, metric, diagnose_msl_alignment(grib_path, config, truth_lat, truth_lon))

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 4 coordinate alignment", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 4 coordinate alignment", gate4)

    def gate5() -> GateResult:
        pipeline = state.get("pipeline") or TyphoonRouteAPipeline(**build_pipeline_kwargs(root, config, args.device))
        tmp_dir = Path(tempfile.mkdtemp(prefix="typhoon_gate_single_"))
        try:
            out = pipeline.run_on_file(str(grib_path), lat_filter=tuple(cfg_get(config, "detection_model", "decode", "lat_filter", default=[0.0, 40.0])))
            out_path = tmp_dir / "all_results.csv"
            out.to_csv(out_path, index=False)
            if out.empty:
                return GateResult("Gate 5 single-file end-to-end", WARN, f"rows=0, output={out_path}")
            has_loc = out[["LAT", "LON", "MSL_MIN"]].notna().all(axis=1).any()
            wind_ok = "PRED_WIND" in out and out["PRED_WIND"].notna().any()
            pres_ok = "PRED_PRES" in out and out["PRED_PRES"].notna().any()
            metric = f"rows={len(out)}, location_ok={has_loc}, wind_non_nan={wind_ok}, pres_non_nan={pres_ok}"
            if has_loc and wind_ok and pres_ok:
                return GateResult("Gate 5 single-file end-to-end", PASS, metric)
            return GateResult("Gate 5 single-file end-to-end", WARN, metric, f"correction_enabled={pipeline.correction_enabled}; inspect history GRIB availability.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 5 single-file end-to-end", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 5 single-file end-to-end", gate5)

    def gate6() -> GateResult:
        if not args.dir:
            return GateResult("Gate 6 directory end-to-end", SKIPPED, "--dir not provided")
        grib_dir = resolve_path(root, args.dir)
        if grib_dir is None or not grib_dir.exists():
            return GateResult("Gate 6 directory end-to-end", FAIL, f"directory not found: {grib_dir}")
        pipeline = state.get("pipeline") or TyphoonRouteAPipeline(**build_pipeline_kwargs(root, config, args.device))
        tmp_dir = Path(tempfile.mkdtemp(prefix="typhoon_gate_dir_"))
        try:
            out = pipeline.run_on_directory(
                aifs_dir=str(grib_dir),
                output_dir=str(tmp_dir),
                lat_filter=tuple(cfg_get(config, "detection_model", "decode", "lat_filter", default=[0.0, 40.0])),
                init_date="latest",
            )
            if out.empty:
                return GateResult("Gate 6 directory end-to-end", WARN, f"rows=0, output={tmp_dir}")
            per_time = out.groupby("ISO_TIME").size().describe().to_dict()
            compact = ", ".join(f"{k}={float(v):.1f}" for k, v in per_time.items())
            return GateResult("Gate 6 directory end-to-end", PASS, f"rows={len(out)}, per_time_stats={{ {compact} }}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 6 directory end-to-end", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 6 directory end-to-end", gate6)

    def gate7() -> GateResult:
        if "detections" not in state:
            return GateResult("Gate 7 localization-only downgrade", SKIPPED, "Gate 3 did not run")
        downgraded = copy.deepcopy(config)
        downgraded.setdefault("correction_model", {})["enabled"] = False
        pipeline = TyphoonRouteAPipeline(**build_pipeline_kwargs(root, downgraded, args.device))
        out = pipeline.run_on_file(str(grib_path), lat_filter=tuple(cfg_get(config, "detection_model", "decode", "lat_filter", default=[0.0, 40.0])))
        if out.empty:
            return GateResult("Gate 7 localization-only downgrade", WARN, "rows=0")
        wind_nan = "PRED_WIND" in out and out["PRED_WIND"].isna().all()
        pres_nan = "PRED_PRES" in out and out["PRED_PRES"].isna().all()
        status = PASS if wind_nan and pres_nan else FAIL
        return GateResult("Gate 7 localization-only downgrade", status, f"rows={len(out)}, wind_all_nan={wind_nan}, pres_all_nan={pres_nan}")

    if gate0_result.status == FAIL:
        results.append(GateResult("Gate 7 localization-only downgrade", SKIPPED, "pygrib missing"))
        print_result(results[-1])
    else:
        run_gate(results, "Gate 7 localization-only downgrade", gate7)

    print()
    print("=" * 72)
    print("Gate summary")
    print("=" * 72)
    for result in results:
        print(f"{result.gate:36s} | {result.status:7s} | {result.metric}")
    failed = [result for result in results if result.status == FAIL]
    if failed:
        print()
        print("Next steps for FAIL items:")
        for result in failed:
            print(f"- {result.gate}: {result.hint or result.metric}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
