#!/bin/bash
set +x

PORT=${1:-8020}

export PYTHON=${PYTHON:-python}

${PYTHON} examples/calvin/main.py --args.port $PORT
