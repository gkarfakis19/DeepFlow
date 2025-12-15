#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASTRA_SIM_DIR="${REPO_ROOT}/astra-sim"

cd "${REPO_ROOT}"

echo "[astrasim] Updating DeepFlow submodules..."
git submodule update --init --recursive
git submodule update --remote -- astra-sim

if [[ ! -d "${ASTRA_SIM_DIR}" ]]; then
  echo "[astrasim] ERROR: expected submodule directory at ${ASTRA_SIM_DIR}" >&2
  exit 1
fi

echo "[astrasim] Updating AstraSim nested submodules..."
git -C "${ASTRA_SIM_DIR}" submodule update --remote -- extern/network_backend/analytical

BUILD_SCRIPT="${ASTRA_SIM_DIR}/build/astra_analytical/build.sh"
if [[ ! -x "${BUILD_SCRIPT}" ]]; then
  echo "[astrasim] ERROR: build script not found or not executable at ${BUILD_SCRIPT}" >&2
  exit 1
fi

echo "[astrasim] Building AstraSim (this may take a while)..."
pushd "${ASTRA_SIM_DIR}" >/dev/null
"${BUILD_SCRIPT}"
popd >/dev/null

echo "[astrasim] AstraSim build completed successfully."
