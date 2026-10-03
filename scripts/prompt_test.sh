#!/bin/bash
# gpt-oss extraction prompt variants on the personality and speech questions (physical, personality, dialogue), each then
# through the archive's exact simplification + entailment steps: low = the archive prompt at low reasoning; brief = the
# personality and speech questions worded as a writer's brief (traits and manner); brieflow = both.
# Usage, inside serve_and_run.py with a gpt-oss server: bash scripts/prompt_test.sh BOOK...
set -e
R=/home/toolkit/chiron_replication/chiron
Q="physical personality dialogue"
run() { python3 -u $R/gen_legacy_gptoss.py --books "${@:3}" --variant $1 --questions $Q $2 --workers 64 && python3 -u $R/gen_legacy_exact.py --books "${@:3}" --variant $1 --workers 48; }
run low "--effort low" "$@" & a=$!
run brief "--brief" "$@" & b=$!
run brieflow "--brief --effort low" "$@" & c=$!
wait $a && wait $b && wait $c
