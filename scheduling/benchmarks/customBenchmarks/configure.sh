#!/bin/sh
set -e

../benchmarkTool/scripts/install_venv.sh
./venv/bin/pip3 install -e ../deps/pythainer/
