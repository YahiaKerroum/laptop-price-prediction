#!/usr/bin/env bash
# Execute every numbered notebook top-to-bottom with papermill.
#
# This is the reproducibility gate: the original notebooks could not run
# end-to-end (04_regression referenced an undefined `best_grid`, and several
# stages read intermediate CSVs that were never committed).
set -uo pipefail

OUT_DIR="${1:-reports/executed}"
mkdir -p "$OUT_DIR"

status=0
shopt -s nullglob
for nb in notebooks/[0-9][0-9]_*.ipynb; do
    name="$(basename "$nb")"
    printf '%-46s' "$name"
    if papermill "$nb" "$OUT_DIR/$name" \
            --kernel python3 --cwd . --no-progress-bar >"$OUT_DIR/${name%.ipynb}.log" 2>&1; then
        echo "OK"
    else
        echo "FAILED  (see $OUT_DIR/${name%.ipynb}.log)"
        tail -n 15 "$OUT_DIR/${name%.ipynb}.log" | sed 's/^/    /'
        status=1
    fi
done

exit "$status"
