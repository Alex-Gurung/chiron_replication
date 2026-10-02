#!/bin/bash
# The Llama notes' pipeline after extraction, with gpt-oss (chiron/gen_legacy_exact.py), then the archive's summary step on
# the result, guided to ~500 words like the Llama summaries (chiron/gen_legsum.py -> outputs/sheets/legsum_x_500).
# Usage, inside serve_and_run.py with a gpt-oss server: bash scripts/legacy_exact.sh BOOK...
set -e
R=/home/toolkit/chiron_replication
python3 -u $R/chiron/gen_legacy_exact.py --books "$@" --workers 64
python3 -u $R/chiron/gen_legsum.py --source legacy_gptoss_x --variant legsum_x_500 --words 500 --books "$@"
