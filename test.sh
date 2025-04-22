#!/bin/bash
set -euo pipefail

uv run pytest -xvs ./synthwave/test_dsl.py
uv run pytest -xvs ./datasetting/test_datasets.py
