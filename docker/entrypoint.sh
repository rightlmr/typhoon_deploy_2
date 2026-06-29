#!/usr/bin/env bash
set -e

cd /app

MODE="${MODE:-once}"
DEVICE="${TC_DEVICE:-cpu}"

case "$MODE" in
  once)
    exec python src/pipeline_routeA.py --config config.yaml --device "$DEVICE" "$@"
    ;;
  watchdog)
    exec python src/watchdog_runner.py --config config.yaml --device "$DEVICE" "$@"
    ;;
  gate)
    exec bash scripts/run_cpu_gate.sh "$@"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    echo "Unknown MODE=$MODE (use once|watchdog|gate|shell)"
    exit 2
    ;;
esac
