#!/bin/bash
set -euo pipefail

uv run pytest -xvs synthwave/test_dsl.py
