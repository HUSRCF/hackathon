#!/usr/bin/env bash
set -euo pipefail

target=${1:?target is required}
max_hits=${2:?max_hits is required}
workers=${3:-8}
pharmer=/root/pharmit-code/src/Release/pharmitserver
data_root=/work/data
query_root=/work/queries
database="$data_root/$target/pharmer-db"
library="$data_root/$target/validation-library.sdf"
queries="$query_root/$target/pharmer-panel"
outputs="$data_root/$target/pharmer-panel-hits"

test -f "$library"
test -f "$queries/panel-receipt.json"
if [[ ! -d "$database" ]]; then
  mkdir -p "$database"
  "$pharmer" -cmd dbcreate -in "$library" -dbdir "$database" -nthreads "$workers"
fi
if [[ -e "$outputs" ]]; then
  echo "refusing to overwrite existing Pharmer output directory: $outputs" >&2
  exit 2
fi
mkdir "$outputs"

export pharmer database outputs max_hits
find "$queries" -maxdepth 1 -type f -name 'q*.json' -print0 |
  sort -z |
  xargs -0 -r -P "$workers" -n 1 bash -c '
    query=$1
    name=${query##*/}
    stem=${name%.json}
    "$pharmer" -q -cmd dbsearch -dbdir "$database" -in "$query" \
      -out "$outputs/$stem.sdf" -max-hits "$max_hits" -max-orient 1 \
      -reduceconfs 1 -sort-rmsd -nthreads 1
  ' bash

actual=$(find "$outputs" -maxdepth 1 -type f -name 'q*.sdf' | wc -l)
expected=$(find "$queries" -maxdepth 1 -type f -name 'q*.json' | wc -l)
if [[ "$actual" -ne "$expected" ]]; then
  echo "Pharmer panel incomplete: expected $expected outputs, observed $actual" >&2
  exit 3
fi
printf '{"target":"%s","query_outputs":%s,"max_hits":%s,"workers":%s}\n' \
  "$target" "$actual" "$max_hits" "$workers"
