#!/bin/bash

if [[ -z "$1" ]]; then
	echo "Usage: ./run.sh <app directory path>"
	exit 1
fi

# change this if you want a different dataset
APPDIR="$1"
DATA_FILE="red.snappy.parquet"

python "$APPDIR/logo_clusterer.py" "$DATA_FILE" \
    --output clusters.json \
    --format json \
    --threshold 0.75 \
    --workers 2 \
    --logos-dir logos \
    --batch-size 50 \
    --delay 1 \
    --chunk-size 100
