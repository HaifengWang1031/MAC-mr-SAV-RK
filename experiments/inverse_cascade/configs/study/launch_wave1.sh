#!/bin/sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=experiments/inverse_cascade/configs/study
L=runs/inverse_cascade_studies/logs
mkdir -p "$L"
nohup sh "$C/run_study.sh" "$L/w1" "$C/mu_0100.json" "$C/mu_0150.json" "$C/mu_0250.json" "$C/mu_0300.json" > "$L/w1.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/w2" "$C/mu_0400.json" "$C/mu_0800.json" "$C/T200_mu0200.json" "$C/T200_mu0800.json" > "$L/w2.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/w3" "$C/seed7_mu0200.json" "$C/seed99_mu0200.json" "$C/seed7_mu0800.json" "$C/seed99_mu0800.json" > "$L/w3.out" 2>&1 &
nohup sh "$C/run_study.sh" "$L/w4" "$C/n512_mu0200_long.json" > "$L/w4.out" 2>&1 &
sleep 5
ps -ax | grep -c "[i]nverse_cascade.run" || true
