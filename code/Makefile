# Makefile — convenience wrappers around scripts/check.sh.
# The canonical entry point is scripts/check.sh; no part of the workflow
# depends on `make` being available (STORY_00_01 open question, resolved
# 2026-05-16).
#
# Usage:
#   make check   — run ruff + black --check + mypy + pytest
#   make lint    — run ruff + black --check only
#   make types   — run mypy only
#   make test    — run pytest only

.PHONY: check lint types test

check:
	bash scripts/check.sh

lint:
	bash scripts/check.sh --lint

types:
	bash scripts/check.sh --types

test:
	bash scripts/check.sh --test
