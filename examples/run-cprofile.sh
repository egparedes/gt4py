#!/bin/bash

set -e

TAG=""
PERF=""
PY_OPTS=""

while getopts 'pt:h' opt; do
  case "$opt" in
    p)
      PERF="true"
      ;;

    t)
      TAG="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p] [-t tag]"
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
  TAG=$(find . -name 'result-*.cprofile*.prof' | wc -l)
fi

OUTPUT="result-${TAG}_${MODE}.cprofile.prof"

cmd="python ${PY_OPTS} -m cProfile -o ${OUTPUT} satadjust3d.py"
echo "Running script: \$ $cmd"
rm -Rf .gt_cache/
$cmd

