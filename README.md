# Typhoon Deploy 2 - Route A

本目录是热力图定位模型与强度订正部署包的集成产物。路线 A 为：

```text
AIFS GRIB2 -> heatmap detect/decode -> per-point intensity correction -> CSV
```

本步不包含 v9 后处理，不输出 `TRACK_ID`，不复制 AIFS GRIB2 数据。

## Directory Sources

- `src/aifs_data_utils.py`: copied from the original deployment package.
- `src/tc_correction_model.py`: copied from the original deployment package.
- `libs/`: copied from the original deployment package for correction inference.
- `checkpoints/correction/`: copied from the original deployment package.
- `tclocator/`: copied from the heatmap locator project and used read-only.
- `checkpoints/detection/finetune_best.ckpt`: copied from the locator training output.
- `checkpoints/detection/norm_stats_aifs.json`: copied from the locator AIFS normalization output.

## GitHub Model Artifacts

The local package contains two checkpoint files, but the GitHub repository
publishes them as Release assets instead of normal Git files:

- `finetune_best.ckpt`
- `epoch.050-val_loss.39.0316.ckpt`

After cloning, download those assets from the repository release page and place
them back at the paths referenced by `config.yaml`.
GitHub normalizes the correction asset name, so rename
`epoch.050-val_loss.39.0316.ckpt` to
`checkpoints/correction/ckpts/epoch=050-val_loss=39.0316.ckpt`.

## Environment

Target runtime:

```text
Ubuntu 22.04
Python 3.10
torch 2.0.1+cu118
libeccodes system package
pygrib
```

Install system GRIB dependencies first, then Python dependencies:

```bash
sudo apt-get update
sudo apt-get install -y libeccodes0 libeccodes-dev
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Put or mount AIFS GRIB2 files under `data/aifs_grib`, or change `data.aifs_grib_dir` in `config.yaml`.

```bash
chmod +x run.sh
./run.sh
```

Equivalent explicit command:

```bash
python src/pipeline_routeA.py --config config.yaml
```

Useful overrides:

```bash
python src/pipeline_routeA.py --config config.yaml --aifs_dir /path/to/grib --output_dir output --init_date latest
python src/pipeline_routeA.py --config config.yaml --lead_min 0 --lead_max 120
python src/pipeline_routeA.py --config config.yaml --device cpu
```

Outputs:

- `output/detections_raw.csv`: decoded heatmap detections before correction.
- `output/all_results.csv`: final route A rows with `LAT`, `LON`, `MSL_MIN`, `CONF`, `PRED_WIND`, `PRED_PRES`.

## Config Notes

- `detection_model.channels`: locator input channels, default `["msl", "vo_850", "t_500"]`.
- `detection_model.norm_stats`: AIFS normalization stats for the locator only.
- `detection_model.decode.conf_thresh`: heatmap confidence threshold, default `0.5`.
- `detection_model.decode.lat_filter`: route A latitude filter.
- `correction_model.*` and `stats.*`: correction checkpoint, config, and correction-only normalization stats.
- `data.init_date`: `null` for all files, `latest` for latest initialization time, or a date string.
- `inference.lead_min` / `lead_max`: optional forecast-hour filter.

All paths in `config.yaml` are relative to this project root, so the same layout can be mounted as `/app`.

## Gate Status

Checked on the local training environment with torch `2.6.0+cu126`, not on the target Linux torch `2.0.1+cu118` environment:

- Python syntax: `py_compile` passed for `src/detector_adapter.py` and `src/pipeline_routeA.py`.
- Import coexistence: `pipeline_routeA` and `tc_correction_model` imported in one process.
- Detection ckpt: raw `torch.load` passed; `HeatmapDetectorAdapter` loaded `finetune_best.ckpt` on CPU.
- Correction ckpt: raw `torch.load` passed.
- Combined initialization: `pipeline_routeA.py --device cpu` loaded both models and exited cleanly with empty `data/aifs_grib`.

Still required on the Linux target server:

- Load both checkpoints with torch `2.0.1+cu118`.
- Run `adapter.predict_from_grib(...)` on one real AIFS GRIB2 and verify the DataFrame contains `ISO_TIME`, `LAT`, `LON`, `MSL_MIN`, `WS`.
- Coordinate gate: compare one known strong-storm short-lead prediction against truth and require roughly less than 80 km error.
- End-to-end single-file route A with non-NaN correction output.
- End-to-end latest-init directory run.
- Correction-missing downgrade check: temporarily point `correction_model.checkpoint` to a missing file and confirm localization still runs with NaN intensity.

## Docker Offline Image

Deferred until all first-step gates pass. The directory is already arranged so the Docker image can use `/app` as project root. The next step should adapt the original Ubuntu 22.04 / Python 3.10 / CUDA 11.8 Dockerfile, set the entrypoint to `python src/pipeline_routeA.py --config config.yaml`, and package with `docker save` for offline delivery.

## Route B TODO

Route B is not implemented here. If track-level continuity becomes necessary, add `tclocator.tracking` between decode and correction, then evaluate storm recall and track precision on months with truth before switching production routing.
