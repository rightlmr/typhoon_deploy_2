# Typhoon Deploy 2 CPU Offline Docker Guide

This guide is for the Linux operator. Codex only authored the files and ran static checks on Windows. Docker build, CPU gate validation, network-off validation, `docker save`, and offline target deployment must be run manually on Linux.

## Scope

The CPU image is intended for x86_64 Linux hosts:

- Build host: online Linux with Docker.
- Target host: offline or LAN-only x86_64 Linux, CPU-only.
- Container project root: `/app`.
- Runtime device: CPU only.

The image includes source code, config, and checkpoint files that are present under `checkpoints/` at build time. It does not download model weights during `docker build`.

## Required Files Before Build

Before building, make sure these files exist in the repository checkout on the online Linux build host:

```text
checkpoints/detection/finetune_best.ckpt
checkpoints/correction/ckpts/epoch=050-val_loss=39.0316.ckpt
checkpoints/correction/fastervit_correction_aifs_historytimestamps_dim64.yaml
checkpoints/correction/stats/normalization_stats.json
checkpoints/correction/stats/ibtracs_stats_correction_log1p.json
checkpoints/detection/norm_stats_aifs.json
```

If the checkpoint files are not in the Git checkout, download them from the GitHub release assets and place them at the paths above before building.

## Build On Online Linux

Run on the online x86_64 Linux build host:

```bash
cd ~/typhoon_deploy_2
docker build -f Dockerfile.cpu -t typhoon_deploy_2:cpu-v0.1 .
```

The default CPU Torch candidate is configured in `Dockerfile.cpu`:

```text
TORCH_VERSION=2.0.1+cpu
TORCHVISION_VERSION=0.15.2+cpu
```

If Gate 1 checkpoint loading fails during CPU validation, rebuild with a different CPU Torch version:

```bash
docker build -f Dockerfile.cpu \
  --build-arg TORCH_VERSION=2.6.0+cpu \
  --build-arg TORCHVISION_VERSION=0.21.0+cpu \
  -t typhoon_deploy_2:cpu-v0.1 .
```

## CPU Gate Validation

Run this on the online Linux build host after preparing a known GRIB case. The GRIB directory must contain any history files required by the selected forecast hour.

Example using the previously validated JANGMI case layout:

```bash
docker run --rm \
  -v /path/to/2026053100:/app/data/aifs_grib \
  -e MODE=gate \
  typhoon_deploy_2:cpu-v0.1 \
  --grib data/aifs_grib/20260531000000-18h-oper-fc.grib2 \
  --truth-lat 23.6 \
  --truth-lon 127.4 \
  --truth-name "JANGMI IBTrACS 2026-05-31 18Z" \
  --dir data/aifs_grib
```

Acceptance guidance for the Linux operator:

- Gate 0 should show `cuda_available=False` inside the CPU container.
- Gate 1 checkpoint load must be `PASS`.
- Gate 4 should be close to the truth point. A strict `PASS` is ideal; a small-distance `WARN` can be accepted by the operator if it matches the chosen case.
- Gate 5 should produce non-NaN wind and pressure for a case with complete history.
- Do not save the offline image until the CPU gate is acceptable for the deployment case.

## Network-Off Runtime Validation

Run this on the online Linux build host after the CPU gate is acceptable:

```bash
docker run --rm --network none \
  -v /path/to/grib:/app/data/aifs_grib \
  -v /path/to/out:/app/output \
  -e MODE=once \
  typhoon_deploy_2:cpu-v0.1 \
  --init_date latest
```

This validates that runtime inference does not require internet access.

## Save Offline Image

Run this on the online Linux build host:

```bash
docker save typhoon_deploy_2:cpu-v0.1 | gzip > typhoon_deploy_2_cpu_v0.1.tar.gz
```

Transfer `typhoon_deploy_2_cpu_v0.1.tar.gz` to the offline target host.

## Load And Run On Offline Target

Run on the offline x86_64 CPU target host:

```bash
gunzip -c typhoon_deploy_2_cpu_v0.1.tar.gz | docker load
```

Single run:

```bash
docker run --rm \
  -v /data/grib:/app/data/aifs_grib \
  -v /data/out:/app/output \
  -e MODE=once \
  typhoon_deploy_2:cpu-v0.1 \
  --init_date latest
```

Interactive shell:

```bash
docker run --rm -it \
  -v /data/grib:/app/data/aifs_grib \
  -v /data/out:/app/output \
  -e MODE=shell \
  typhoon_deploy_2:cpu-v0.1
```

## Watchdog Mode

Start the watchdog in the background:

```bash
docker run -d --name tc_watch \
  -v /data/grib:/app/data/aifs_grib \
  -v /data/out:/app/output \
  -e MODE=watchdog \
  typhoon_deploy_2:cpu-v0.1 \
  --poll-interval 300
```

View logs:

```bash
docker logs -f tc_watch
```

View status JSON:

```bash
docker exec tc_watch python src/watchdog_runner.py --status
```

Stop and remove:

```bash
docker stop tc_watch
docker rm tc_watch
```

The watchdog writes:

```text
output/watchdog_status.json
output/processed.log
output/detections_raw.csv
output/all_results.csv
```

`watchdog_status.json` contains:

```text
state
last_poll_time
processed_files_count
last_processed
pending_count
poll_interval_sec
errors_recent
```

## Runtime Modes

The container entrypoint dispatches by `MODE`:

```text
MODE=once      python src/pipeline_routeA.py --config config.yaml --device cpu
MODE=watchdog  python src/watchdog_runner.py --config config.yaml --device cpu
MODE=gate      bash scripts/run_cpu_gate.sh
MODE=shell     /bin/bash
```

`TC_DEVICE` defaults to `cpu`; the Dockerfile also sets `CUDA_VISIBLE_DEVICES=""`.

## Linux Validation Table

Fill this table after running the commands on Linux.

| Item | Command Section | Expected | Actual Result |
|---|---|---|---|
| CPU environment | CPU Gate Validation | `cuda_available=False` | |
| Gate 1 checkpoint load | CPU Gate Validation | PASS | |
| Gate 4 coordinate alignment | CPU Gate Validation | PASS or acceptable WARN for known truth | |
| Gate 5 CPU end-to-end | CPU Gate Validation | wind and pressure non-NaN | |
| Network-off once | Network-Off Runtime Validation | succeeds with `--network none` | |
| Docker save/load | Save Offline Image / Load And Run | target can load image | |
| Watchdog start | Watchdog Mode | container keeps polling | |
| Watchdog status | Watchdog Mode | `watchdog_status.json` is readable | |
| Watchdog stop | Watchdog Mode | state becomes `stopped` | |

## Troubleshooting

- If Gate 1 fails, try a different CPU Torch version and rebuild.
- If Gate 5 outputs NaN intensity, check that the mounted GRIB directory contains the history files required by the selected forecast hour.
- If `pygrib` import fails during build or runtime, check `libeccodes-dev`, `libeccodes-data`, and `libproj-dev` availability on the build host.
- If the container accidentally sees a GPU, confirm `CUDA_VISIBLE_DEVICES=""` and `TC_DEVICE=cpu` are set.
