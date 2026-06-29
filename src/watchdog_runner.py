"""Foreground watchdog runner for the Route A Docker container."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline_routeA import TyphoonRouteAPipeline, load_config_from_yaml


DEFAULT_POLL_INTERVAL_SEC = 300
MAX_RECENT_ERRORS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def cfg_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_processed(path: Path, files: list[str]) -> None:
    if not files:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for item in files:
            f.write(f"{item}\n")


def write_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "state": "not_started",
            "last_poll_time": None,
            "processed_files_count": 0,
            "last_processed": None,
            "pending_count": None,
            "poll_interval_sec": None,
            "errors_recent": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(root: Path, config: Mapping[str, Any], aifs_dir: Path, device: str) -> TyphoonRouteAPipeline:
    det_cfg = dict(config.get("detection_model", {}))
    corr_cfg = dict(config.get("correction_model", {}))
    stats_cfg = dict(config.get("stats", {}))

    return TyphoonRouteAPipeline(
        detection_model_dir=str(resolve_path(root, det_cfg.get("model_dir", "checkpoints/detection"))),
        detection_config=det_cfg,
        correction_ckpt=str(resolve_path(root, corr_cfg.get("checkpoint"))) if corr_cfg.get("checkpoint") else None,
        correction_config=str(resolve_path(root, corr_cfg.get("config"))) if corr_cfg.get("config") else None,
        aifs_stats_file=str(resolve_path(root, stats_cfg.get("aifs_stats"))) if stats_cfg.get("aifs_stats") else None,
        ibtracs_stats_file=str(resolve_path(root, stats_cfg.get("ibtracs_stats"))) if stats_cfg.get("ibtracs_stats") else None,
        aifs_grib_dir=str(aifs_dir),
        correction_enabled=bool(corr_cfg.get("enabled", True)),
        device=device,
    )


def build_status(
    state: str,
    processed: set[str],
    pending_count: int,
    poll_interval: int,
    errors_recent: list[dict[str, str]],
    last_processed: str | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "last_poll_time": utc_now(),
        "processed_files_count": len(processed),
        "last_processed": last_processed,
        "pending_count": pending_count,
        "poll_interval_sec": poll_interval,
        "errors_recent": errors_recent[-MAX_RECENT_ERRORS:],
    }


def process_once(
    pipeline: TyphoonRouteAPipeline,
    config: Mapping[str, Any],
    aifs_dir: Path,
    output_dir: Path,
    processed_log: Path,
    status_path: Path,
    poll_interval: int,
    errors_recent: list[dict[str, str]],
) -> int:
    processed = load_processed(processed_log)
    data_cfg = dict(config.get("data", {}))
    selected_files = pipeline._select_grib_files(str(aifs_dir), data_cfg.get("init_date"))
    pending = [path for path in selected_files if path not in processed]

    write_status(
        status_path,
        build_status("idle" if not pending else "processing", processed, len(pending), poll_interval, errors_recent),
    )

    if not pending:
        print(f"[Watchdog] {utc_now()} no new GRIB files")
        return 0

    print(f"[Watchdog] {utc_now()} processing {len(pending)} new GRIB file(s)")
    det_cfg = dict(config.get("detection_model", {}))
    inf_cfg = dict(config.get("inference", {}))
    decode_cfg = dict(det_cfg.get("decode", {}))
    cfg_lat_filter = decode_cfg.get("lat_filter", [0.0, 40.0])

    try:
        pipeline.run_on_directory(
            aifs_dir=str(aifs_dir),
            output_dir=str(output_dir),
            patch_size=int(det_cfg.get("patch_size", 40)),
            eps=float(det_cfg.get("eps", 0.1)),
            lat_filter=(float(cfg_lat_filter[0]), float(cfg_lat_filter[1])),
            lead_min=inf_cfg.get("lead_min"),
            lead_max=inf_cfg.get("lead_max"),
            init_date=data_cfg.get("init_date"),
        )
    except Exception as exc:
        print(f"[Watchdog] ERROR: {exc}")
        traceback.print_exc()
        errors_recent.append({
            "time": utc_now(),
            "error": str(exc),
        })
        write_status(
            status_path,
            build_status("error", processed, len(pending), poll_interval, errors_recent),
        )
        return 1

    append_processed(processed_log, pending)
    processed = load_processed(processed_log)
    last_processed = pending[-1] if pending else None
    write_status(
        status_path,
        build_status("idle", processed, 0, poll_interval, errors_recent, last_processed),
    )
    print(f"[Watchdog] {utc_now()} completed; processed_files_count={len(processed)}")
    return 0


def sleep_until_next_poll(seconds: int, should_stop: dict[str, bool]) -> None:
    deadline = time.time() + seconds
    while not should_stop["value"] and time.time() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.time())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Route A watchdog runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=os.environ.get("TC_DEVICE", "cpu"))
    parser.add_argument("--poll-interval", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Process one poll cycle and exit")
    parser.add_argument("--status", action="store_true", help="Print watchdog status JSON and exit")
    parser.add_argument("--aifs-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    root = config_path.parent

    if config_path.exists():
        config = load_config_from_yaml(config_path)
    else:
        config = {}

    data_cfg = dict(config.get("data", {}))
    output_cfg = dict(config.get("output", {}))
    watchdog_cfg = dict(config.get("watchdog", {}))

    aifs_dir = resolve_path(root, args.aifs_dir or data_cfg.get("aifs_grib_dir", "data/aifs_grib"))
    output_dir = resolve_path(root, args.output_dir or output_cfg.get("output_dir", "output"))
    if aifs_dir is None or output_dir is None:
        raise ValueError("AIFS and output directories must be configured")

    status_path = output_dir / "watchdog_status.json"
    processed_log = output_dir / "processed.log"
    poll_interval = int(args.poll_interval or watchdog_cfg.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))

    if args.status:
        print(json.dumps(read_status(status_path), indent=2, sort_keys=True))
        return 0

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    should_stop = {"value": False}
    errors_recent: list[dict[str, str]] = []

    def request_stop(signum, _frame) -> None:
        print(f"[Watchdog] received signal {signum}; stopping after current step")
        should_stop["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    output_dir.mkdir(parents=True, exist_ok=True)
    aifs_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(root, config, aifs_dir, args.device)

    try:
        while not should_stop["value"]:
            process_once(
                pipeline=pipeline,
                config=config,
                aifs_dir=aifs_dir,
                output_dir=output_dir,
                processed_log=processed_log,
                status_path=status_path,
                poll_interval=poll_interval,
                errors_recent=errors_recent,
            )
            if args.once:
                break
            sleep_until_next_poll(poll_interval, should_stop)
    finally:
        processed = load_processed(processed_log)
        if aifs_dir.exists():
            selected_files = pipeline._select_grib_files(str(aifs_dir), data_cfg.get("init_date"))
        else:
            selected_files = []
        pending_count = len([path for path in selected_files if path not in processed])
        write_status(
            status_path,
            build_status("stopped", processed, pending_count, poll_interval, errors_recent),
        )
        print("[Watchdog] stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
