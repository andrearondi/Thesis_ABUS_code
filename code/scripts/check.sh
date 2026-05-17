#!/usr/bin/env bash
# check.sh — Local quality gate for the abus package.
#
# Usage:
#   scripts/check.sh            # run all checks
#   scripts/check.sh --lint     # ruff + black only
#   scripts/check.sh --types    # mypy only
#   scripts/check.sh --test     # pytest only
#
# Exit code: non-zero on any failure.
# This script is the canonical entry point; the Makefile targets are thin
# wrappers. No part of the workflow depends on `make` being available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT}"

RUN_LINT=true
RUN_TYPES=true
RUN_TEST=true

for arg in "$@"; do
    case "$arg" in
        --lint)  RUN_TYPES=false; RUN_TEST=false ;;
        --types) RUN_LINT=false;  RUN_TEST=false ;;
        --test)  RUN_LINT=false;  RUN_TYPES=false ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

RUFF_STATUS=0
BLACK_STATUS=0
MYPY_STATUS=0
PYTEST_STATUS=0

if $RUN_LINT; then
    echo "==> ruff check ..."
    ruff check src tests scripts || RUFF_STATUS=$?
    echo "==> black --check ..."
    black --check src tests scripts || BLACK_STATUS=$?
fi

if $RUN_TYPES; then
    echo "==> mypy src ..."
    mypy src || MYPY_STATUS=$?
fi

if $RUN_TEST; then
    echo "==> pytest ..."
    pytest || PYTEST_STATUS=$?
fi

echo ""
echo "========================================"

ruff_label="OK"
black_label="clean"
mypy_label="0 errors"
pytest_label="passed"

if $RUN_LINT; then
    [ $RUFF_STATUS -ne 0 ] && ruff_label="FAILED (${RUFF_STATUS})"
    [ $BLACK_STATUS -ne 0 ] && black_label="FAILED (${BLACK_STATUS})"
    echo "ruff:  ${ruff_label} | black: ${black_label}"
fi
if $RUN_TYPES; then
    [ $MYPY_STATUS -ne 0 ] && mypy_label="FAILED (${MYPY_STATUS})"
    echo "mypy:  ${mypy_label}"
fi
if $RUN_TEST; then
    [ $PYTEST_STATUS -ne 0 ] && pytest_label="FAILED (${PYTEST_STATUS})"
    echo "pytest: ${pytest_label}"
fi

echo "========================================"

OVERALL=$((RUFF_STATUS + BLACK_STATUS + MYPY_STATUS + PYTEST_STATUS))
if [ $OVERALL -ne 0 ]; then
    echo "GATE: FAILED"
    exit 1
fi
echo "GATE: OK"
