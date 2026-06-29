#!/usr/bin/env bash
set -e

cd /app

python scripts/gate_check.py --config config.yaml --device cpu "$@"
