#!/bin/sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=experiments/inverse_cascade/configs/study
L=runs/inverse_cascade_studies/logs
mkdir -p "$L"
nohup sh "$C/run_study.sh" "$L/p1" "$C/mu_0200.json"            > "$L/p1.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/p2" "$C/q_192_dt005.json"        > "$L/p2.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/p3" "$C/q_128_dt005.json"        > "$L/p3.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/p4" "$C/q_192_dt0025.json"       > "$L/p4.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/p5" "$C/n512_mu0200_long.json"   > "$L/p5.out" 2>&1 &
sleep 5
ps -ax | grep -c "[i]nverse_cascade.run" || true
