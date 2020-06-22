#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source $ROOT/.anaconda/bin/activate

set -e
set -x

# test
pip show hacking &>/dev/null || pip install -q hacking
flake8 $ROOT

set +x
set +e
