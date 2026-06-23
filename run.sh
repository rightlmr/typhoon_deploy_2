#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python src/pipeline_routeA.py --config config.yaml
