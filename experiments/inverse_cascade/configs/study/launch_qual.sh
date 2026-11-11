#!/bin/sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=experiments/inverse_cascade/configs/study
L=runs/inverse_cascade_studies/logs
mkdir -p "$L"
nohup sh "$C/run_study.sh" "$L/q1" "$C/q_192_dt005.json" "$C/q_192_dt0025.json" > "$L/q1.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/q2" "$C/q_128_dt005.json" > "$L/q2.out" 2>&1 &
sleep 3
ps -ax | grep -c "[i]nverse_cascade.run" || true
