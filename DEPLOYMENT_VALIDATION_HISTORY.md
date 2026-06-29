# Typhoon Deploy 2 Conversation And Validation History

Last updated: 2026-06-29

## Project Outcome

The localization project from `F:\typhoon_loc` and the deployment project from `F:\typhoon_deploy` were integrated into `F:\typhoon_deploy_2`.

The current Route A deployment flow is:

1. Read ECMWF AIFS GRIB2 fields.
2. Run the heatmap localization model.
3. Optionally load historical AIFS fields.
4. Run the intensity correction model.
5. Export route A detection and correction results.
6. Downgrade cleanly to localization-only output when correction is disabled or history is unavailable.

GitHub repository:

- Public repository: <https://github.com/rightlmr/typhoon_deploy_2>
- Release tag: <https://github.com/rightlmr/typhoon_deploy_2/releases/tag/v0.1.0>
- Detection checkpoint release asset: `finetune_best.ckpt`
- Correction checkpoint release asset: `epoch.050-val_loss.39.0316.ckpt`
- Local correction checkpoint path after download/rename: `checkpoints/correction/ckpts/epoch=050-val_loss=39.0316.ckpt`

## Important Files

- `config.yaml`: main deployment configuration.
- `run.sh`: simple Linux entrypoint.
- `src/detector_adapter.py`: adapter around the heatmap locator.
- `src/pipeline_routeA.py`: route A end-to-end pipeline.
- `src/aifs_data_utils.py`: AIFS field extraction for correction input.
- `tclocator/io_aifs.py`: AIFS field extraction for localization input.
- `scripts/gate_check.py`: integration gate check added from `ADD_GATE_CHECK.md`.
- `DEPLOYMENT_VALIDATION_HISTORY.md`: this conversation and validation summary.

## Linux Setup Notes

The first Linux server was `aarch64` with NVIDIA GB10. That path was harder because the deployment requirements were based on `torch==2.0.1+cu118`, which is x86_64 CUDA-oriented.

The simpler path was to use an x86_64 NVIDIA CUDA Linux server.

Observed issues and resolutions:

- Ubuntu 24.04 does not provide `python3.10-venv` from the default Noble repository.
- Conda `defaults`/`repo.anaconda.com` returned HTTP 403 in the server environment.
- The working setup used a conda Python 3.10 environment and conda-forge/mirror configuration.
- Server validation environment eventually passed with:
  - `torch=2.0.1+cu118`
  - `cuda_available=True`
  - `pygrib=True`

## Gate Check Specification

`scripts/gate_check.py` checks:

- Gate 0: environment, Torch/CUDA/pygrib.
- Gate 1: detection and correction checkpoint loading.
- Gate 2: detector/correction dependency coexistence.
- Gate 3: detector interface on one GRIB file.
- Gate 4: coordinate alignment against a known truth point.
- Gate 5: single-file end-to-end pipeline, including correction output.
- Gate 6: directory end-to-end pipeline.
- Gate 7: localization-only downgrade.

## AIFS Filename And Directory Notes

Supported AIFS file names look like:

```text
YYYYMMDDHHMMSS-Hh-oper-fc.grib2
```

Example:

```text
20260531000000-18h-oper-fc.grib2
```

The `.index` files can exist beside `.grib2`, but the pipeline reads the `.grib2` files.

For Gate 5, the history lookup uses `data.aifs_grib_dir` from the selected config file, not only the `--dir` argument. Therefore, when validating data stored outside `data/aifs_grib`, create a temporary config and point `data.aifs_grib_dir` to the real data directory.

Example for server data in `data/2026053100`:

```bash
cd ~/typhoon_deploy_2
cp config.yaml config.2026053100.yaml

python - <<'PY'
from pathlib import Path
p = Path("config.2026053100.yaml")
s = p.read_text()
s = s.replace('aifs_grib_dir: "data/aifs_grib"', 'aifs_grib_dir: "data/2026053100"')
p.write_text(s)
PY
```

## GRIB Longitude Bug And Fix

Initial server validation produced an obviously wrong longitude:

```text
pred_top=(23.500,306.500)
dist_to_truth=15500+ km
```

The cause was that AIFS GRIB values were being interpreted against fixed 0..360 longitude arrays without first respecting the latitude/longitude order returned by the GRIB message.

Fix applied:

- `tclocator/io_aifs.py`
- `src/aifs_data_utils.py`

Both readers now use `message.latlons()` to canonicalize global GRIB fields to:

- latitude: north to south
- longitude: 0..360 increasing

After the fix, the same case moved from `306.5E` to `126.5E`, matching the western North Pacific system.

## Truth Data Used

Truth source:

- NOAA/NCEI IBTrACS v04r01
- CSV directory: <https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/>
- Local file used during validation: `data/truth/ibtracs.last3years.list.v04r01.csv`

Important point: the originally supplied truth point `16.2N, 131.5E` was not valid for the `20260531120000-6h` effective time. It matched JANGMI around `2026-05-29 12Z`, not `2026-05-31 18Z`.

IBTrACS JANGMI positions for the `20260531000000` cycle:

```text
20260531000000-0h  -> 2026-05-31 00Z: 20.9N, 128.1E
20260531000000-6h  -> 2026-05-31 06Z: 21.8N, 127.8E
20260531000000-12h -> 2026-05-31 12Z: 22.8N, 127.7E
20260531000000-18h -> 2026-05-31 18Z: 23.6N, 127.4E
```

IBTrACS JANGMI positions for the `20260531120000` cycle:

```text
20260531120000-6h  -> 2026-05-31 18Z: 23.6N, 127.4E
20260531120000-18h -> 2026-06-01 06Z: 25.6N, 127.4E
```

## Validation Data Downloaded Locally

The local Windows validation downloaded AIFS files to:

```text
F:\typhoon_deploy_2\data\aifs_grib
```

Downloaded files:

```text
20260531120000-0h-oper-fc.grib2
20260531120000-6h-oper-fc.grib2
20260531120000-12h-oper-fc.grib2
20260531120000-18h-oper-fc.grib2
```

ECMWF Open Data S3 path pattern:

```text
https://ecmwf-forecasts.s3.amazonaws.com/YYYYMMDD/HHz/aifs-single/0p25/oper/YYYYMMDDHH0000-FHh-oper-fc.grib2
```

Example:

```text
https://ecmwf-forecasts.s3.amazonaws.com/20260531/12z/aifs-single/0p25/oper/20260531120000-18h-oper-fc.grib2
```

Downloaded GRIB and truth CSV files are local validation artifacts and should not be committed.

## Local Validation Results

Local environment:

```text
Python 3.11
torch 2.6.0+cu126
pygrib 2.1.8
numpy 1.26.4
```

For `20260531120000-6h-oper-fc.grib2` with truth `23.6N,127.4E`:

```text
Gate 3 PASS: pred=(23.5N,126.5E)
Gate 4 WARN: dist=92.4 km
Gate 5 WARN: history missing for single-file correction
Gate 6 PASS
Gate 7 PASS
```

For `20260531120000-18h-oper-fc.grib2` with truth `25.6N,127.4E`:

```text
Gate 0 PASS
Gate 1 PASS
Gate 2 PASS
Gate 3 PASS: pred=(25.5N,127.5E)
Gate 4 PASS: dist=15.0 km
Gate 5 PASS: wind_non_nan=True, pres_non_nan=True
Gate 6 PASS
Gate 7 PASS
Correction output: wind=95.0 kt, pres=927.3 hPa
```

This confirmed that localization, GRIB reading, history loading, correction, directory processing, and localization-only downgrade all work.

## Server Validation Results

The server initially used data under:

```text
~/typhoon_deploy_2/data/2026053100
```

After creating `config.2026053100.yaml` with:

```yaml
data:
  aifs_grib_dir: "data/2026053100"
```

the final server gate summary was:

```text
Gate 0 environment                   | PASS | torch=2.0.1+cu118, cuda_available=True, pygrib=True
Gate 1 checkpoint load               | PASS
Gate 2 dependency coexistence        | PASS | device=cuda, use_history=True, history=[6, 12, 18]
Gate 3 detector interface            | PASS | rows=1, top=(lat=23.500, lon=126.500, conf=0.6207, msl_min=97303.0)
Gate 4 coordinate alignment          | WARN | truth=(23.600,127.400), pred_top=(23.500,126.500), dist=92.4km
Gate 5 single-file end-to-end        | PASS | rows=1, location_ok=True, wind_non_nan=True, pres_non_nan=True
Gate 6 directory end-to-end          | PASS | rows=32, per_time_stats={ count=27.0, mean=1.2, std=0.4, min=1.0, 25%=1.0, 50%=1.0, 75%=1.0, max=2.0 }
Gate 7 localization-only downgrade   | PASS | rows=1, wind_all_nan=True, pres_all_nan=True
```

Interpretation:

- The pipeline is deployed and usable.
- Gate 4 WARN at `92.4 km` is a strict-threshold precision warning, not a broken pipeline.
- Gate 5 PASS confirms history GRIB loading and intensity correction are working.
- Gate 6 PASS confirms directory-level batch processing works.

## Reproducible Server Command

Use this command for the `2026053100` directory:

```bash
cd ~/typhoon_deploy_2

python scripts/gate_check.py \
  --config config.2026053100.yaml \
  --grib data/2026053100/20260531000000-18h-oper-fc.grib2 \
  --truth-lat 23.6 \
  --truth-lon 127.4 \
  --truth-name "JANGMI IBTrACS 2026-05-31 18Z" \
  --dir data/2026053100 \
  --device cuda
```

Before running, verify that the four history files exist:

```bash
ls -lh data/2026053100/20260531000000-{0,6,12,18}h-oper-fc.grib2
```

## Operational Lessons

1. For single-file correction, the selected config must point `data.aifs_grib_dir` to the directory containing the history files.
2. `--dir` is for Gate 6 directory validation and does not replace `data.aifs_grib_dir` for Gate 5 history lookup.
3. Use `18h` or later leads when validating correction with same-init history, because the `6/12/18h` history fields can be satisfied by `0h/6h/12h` files.
4. If validating `6h`, previous-day history files may be required for older target forecast hours.
5. Gate 4 requires truth for the GRIB effective time, not the initial time.
6. A Gate 4 WARN can still be acceptable for deployment if the distance is close to the strict threshold and all end-to-end gates pass.
