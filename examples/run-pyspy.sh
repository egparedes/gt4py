#!/bin/bash

set -e

TAG=""
PERF=""
PY_OPTS=""
SAMPLING="100"

while getopts 'ps:t:h' opt; do
  case "$opt" in
    p)
      PERF="true"
      ;;

    t)
      TAG="$OPTARG"
      ;;
   
    s)
      SAMPLING="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p] [-s samples_per_sec] [-t tag]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

if [[ -z "$PERF" ]]; then
  MODE="debug"
  PY_OPTS="${PY_OPTS}"
else
  MODE="OO"
  PY_OPTS="${PY_OPTS} -OO"
fi

if [[ -z "$TAG" ]]; then
  TAG=$(find . -name 'result-*.pyspy-*.json' | wc -l)
fi

OUTPUT="result-${TAG}_${MODE}.pyspy-${SAMPLING}.json "

cmd="py-spy record -o ${OUTPUT} -f speedscope -- python ${PY_OPTS} satadjust3d.py"
echo "Running script: \$ $cmd"
rm -Rf .gt_cache/
$cmd

