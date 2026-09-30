#!/bin/bash
# Re-run every analysis the report reads, then render reports/results.html.
# Usage: bash scripts/final_analysis.sh   (from anywhere; plain python3, no GPU)
# Dense windows use books with >= 5 passages (few books have 10); every other set uses >= 10.
set -euo pipefail
cd "$(dirname "$0")/../chiron"
for m in Qwen3-4B-Instruct-2507 Qwen3.5-9B-Base Qwen3.8-27B_nothink Qwen3.8-27B_think Mistral-7B-Instruct-v0.2_prefix; do
  python3 analyze.py --set main --model $m > /dev/null
done
for m in Qwen3-4B-Instruct-2507 Qwen3.5-9B-Base Qwen3.8-27B_nothink Qwen3.8-27B_think; do
  python3 analyze.py --set short --model $m > /dev/null
  python3 analyze.py --set window --model $m --min-items 5 > /dev/null
done
python3 analyze.py --set two --model Qwen3-4B-Instruct-2507 > /dev/null
python3 analyze.py --set pron --model Qwen3-4B-Instruct-2507 > /dev/null
python3 analyze_items.py --model Qwen3.8-27B_nothink > /dev/null
python3 analyze_items.py --model Qwen3.5-9B-Base > /dev/null
python3 analyze_gender.py > /dev/null
python3 analyze_ppl.py > /dev/null
python3 analyze_traces.py > /dev/null
python3 analyze_manual.py > /dev/null
python3 make_report.py
