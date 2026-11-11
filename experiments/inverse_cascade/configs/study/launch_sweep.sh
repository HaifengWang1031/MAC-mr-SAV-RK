#!/bin/sh
# Launch the mu sweep (T*=100) on four sequential workers.
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=experiments/inverse_cascade/configs/study
L=runs/inverse_cascade_studies/logs
mkdir -p "$L"
launch() { nohup sh "$C/run_study.sh" "$L/$1" "$2" "$3" > "$L/$1.out" 2>&1 & }
launch s1 "$C/mu_0050.json" "$C/mu_0100.json"
launch s2 "$C/mu_0150.json" "$C/mu_0200.json"
launch s3 "$C/mu_0250.json" "$C/mu_0300.json"
launch s4 "$C/mu_0400.json" "$C/mu_0800.json"
sleep 3
ps -ax | grep -c "[i]nverse_cascade.run" || true
