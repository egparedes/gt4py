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
  TAG=$(find . -name 'result-*.time.txt' | wc -l)
fi

OUTPUT="result-${TAG}_${MODE}.time.txt"

cmd="time python ${PY_OPTS} satadjust3d.py"
echo "Running script: \$ $cmd > ${OUTPUT}"
rm -Rf .gt_cache/
$cmd &> ${OUTPUT} 
