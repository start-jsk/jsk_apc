#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source $ROOT/.anaconda/bin/activate

set -e
set -x

# test
pip install -q pytest
pytest -v $ROOT/tests -m 'not slow'

set +x
set +e
