#!/bin/bash
# Every representation from one generator for some books, on this job's server. CHIRON_GEN picks the generator
# (q4b = Qwen3-4B-Instruct, l70 = Llama-3.3-70B; gpt-oss ran these scripts one by one earlier). Stages run side by side:
#   chiron  CHIRON notes with the archive's pipeline (extraction, simplification, entailment) and its condensed sheet
#   summ    rolling character summaries (gen_summary.py)
#   csum    the CHIRON paper's character summary: whole story so far -> summary, entailment-filtered (gen_csum.py)
#   plot    plot summaries, hierarchical and global (gen_plot.py)
# Usage, inside serve_and_run.py: CHIRON_GEN=q4b bash scripts/gen_pipeline.sh "chiron summ csum plot" BOOK...
set -u
R=/home/toolkit/chiron_replication/chiron; G=$CHIRON_GEN; STAGES=$1; shift
pids=()
for s in $STAGES; do
  case $s in
    chiron) (python3 -u $R/gen_legacy_gptoss.py --books "$@" --workers 64 && python3 -u $R/gen_legacy_exact.py --books "$@" --workers 48 \
             && python3 -u $R/gen_legsum.py --source exact --variant legsum_${G}_x --books "$@") & ;;
    summ) python3 -u $R/gen_summary.py --books "$@" --tag "$1" & ;;
    csum) python3 -u $R/gen_csum.py --books "$@" & ;;
    csumraw) python3 -u $R/gen_csum.py --no-filter --books "$@" & ;;
    plot) (python3 -u $R/gen_plot.py hier --books "$@" && python3 -u $R/gen_plot.py global --books "$@") & ;;
  esac
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
