#!/bin/bash
set -euo pipefail

build_dir=${BUILD_DIR:-build}
cmake -G Ninja -S . -B "$build_dir" "$@"
cmake --build "$build_dir" --parallel "${JOBS:-2}"

echo "Build complete: $build_dir/spirula-sfm"
