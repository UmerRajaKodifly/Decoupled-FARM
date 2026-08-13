#!/usr/bin/env bash
# Back-compat wrapper — use scripts/run_experiment.sh directly.
exec "$(dirname "$0")/run_experiment.sh" "$@"
