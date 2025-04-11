#!/bin/bash
set -euo pipefail

pytest -xvs synthwave/test_dsl.py
