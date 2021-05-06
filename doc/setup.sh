#!/bin/sh

type virtualenv &>/dev/null || {
  echo "Please install virtualenv: pip install virtualenv" >&2
  return 1
}

unset PYTHONPATH
virtualenv --python=python3 venv
. venv/bin/activate
pip install -r requirements.txt
