#!/usr/bin/env bash
# Execute every numbered notebook top-to-bottom with papermill.
#
# This is the reproducibility gate: the original notebooks could not run
# end-to-end (04_regression referenced an undefined `best_grid`, and several
# stages read intermediate CSVs that were never committed).
#
# FAST mode (the default here) shrinks notebook 04's hyperparameter searches.
# The original searches are 576 LightGBM candidates at 5 folds - 2,880 fits -
# plus 250 for Random Forest and two XGBoost grids, which takes hours and makes
# the gate unusable. FAST changes nothing but the size of those searches.
#
#   ./scripts/run_notebooks.sh                 # fast, for verification
#   ./scripts/run_notebooks.sh --full          # the original searches
set -uo pipefail

FAST=true
OUT_DIR="reports/executed"

for arg in "$@"; do
    case "$arg" in
        --full) FAST=false ;;
        --fast) FAST=true ;;
        *) OUT_DIR="$arg" ;;
    esac
done

mkdir -p "$OUT_DIR"
echo "hyperparameter searches: $([ "$FAST" = true ] && echo 'FAST (use --full for the original)' || echo 'FULL - this takes hours')"
echo

status=0
shopt -s nullglob
for nb in notebooks/[0-9][0-9]_*.ipynb; do
    name="$(basename "$nb")"
    log="$OUT_DIR/${name%.ipynb}.log"
    printf '%-46s' "$name"
    start=$SECONDS

    if papermill "$nb" "$OUT_DIR/$name" \
            -p FAST "$FAST" \
            --kernel python3 --cwd . --no-progress-bar >"$log" 2>&1; then
        printf 'OK   %4ds\n' "$((SECONDS - start))"
    else
        printf 'FAILED  (see %s)\n' "$log"
        tail -n 15 "$log" | sed 's/^/    /'
        status=1
    fi
done

exit "$status"
