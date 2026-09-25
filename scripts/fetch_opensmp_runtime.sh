#!/usr/bin/env bash
# Fetch the GNU OpenMP runtime that LightGBM links against.
#
# Why this exists
# ---------------
# LightGBM's manylinux wheel is linked against libgomp.so.1 (GNU OpenMP). This WSL
# image does not ship it, and there is no root access, so `apt-get install libgomp1` is
# not an option. Importing lightgbm therefore fails with:
#
#     OSError: libgomp.so.1: cannot open shared object file: No such file or directory
#
# The fix is to extract the official Ubuntu `libgomp1` package into `.vendor/libgomp/`
# and let `src/models/lightgbm_model.py` preload it into the global symbol namespace
# only when the plain import has already failed. A machine with a system OpenMP runtime
# keeps using its own copy; this is a fallback, not an override.
#
# The vendored copy is committed so that `pytest tests/ -v` works on a fresh clone with
# no system changes at all.
#
# Usage:
#   bash scripts/fetch_opensmp_runtime.sh
#
# Verified on: Ubuntu 24.04 (noble), x86_64, lightgbm 4.7.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/.vendor/libgomp"
ARCHIVE_BASE="http://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14"
DEB_NAME="libgomp1_14.2.0-4ubuntu2~24.04.1_amd64.deb"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "[1/3] locating ${DEB_NAME}"
if ! curl -sIL "${ARCHIVE_BASE}/${DEB_NAME}" | grep -q "HTTP/.* 200"; then
    echo "ERROR: ${DEB_NAME} not found at ${ARCHIVE_BASE}." >&2
    echo "Browse that directory for the current libgomp1 version and update DEB_NAME." >&2
    exit 1
fi

echo "[2/3] downloading"
curl -sL -o "${WORK_DIR}/libgomp1.deb" "${ARCHIVE_BASE}/${DEB_NAME}"

echo "[3/3] extracting into ${VENDOR_DIR}"
dpkg-deb -x "${WORK_DIR}/libgomp1.deb" "${WORK_DIR}/extracted"
mkdir -p "${VENDOR_DIR}"
cp "${WORK_DIR}/extracted/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0" "${VENDOR_DIR}/"
ln -sf libgomp.so.1.0.0 "${VENDOR_DIR}/libgomp.so.1"

echo "done. verifying..."
VENDOR_LIB="${VENDOR_DIR}/libgomp.so.1" "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import ctypes
import os

lib = os.environ["VENDOR_LIB"]
ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
import lightgbm

print(f"lightgbm {lightgbm.__version__} imports with vendored OpenMP runtime at {lib}")
PY
