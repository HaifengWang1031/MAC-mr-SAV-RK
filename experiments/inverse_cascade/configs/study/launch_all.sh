#!/bin/sh
# Launch up to six independent sequential workers for the inverse-cascade study.
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=experiments/inverse_cascade/configs/study
L=runs/inverse_cascade_studies/logs
mkdir -p "$L"
launch() { nohup sh "$C/run_study.sh" "$L/$1" "$2" "$3" > "$L/$1.out" 2>&1 & }
launch q1 "$C/qual_256_dt005.json"  "$C/qual_256_dt0025.json"
launch q2 "$C/qual_192_dt005.json"  "$C/qual_128_dt005.json"
launch q3 "$C/mu_0050.json"         "$C/mu_0100.json"
launch q4 "$C/mu_0150.json"         "$C/mu_0250.json"
launch q5 "$C/mu_0300.json"         "$C/mu_0400.json"
launch q6 "$C/mu_0200.json"         "$C/mu_0800.json"
sleep 3
ps -o pid,etime,command -ax | grep -c "[i]nverse_cascade.run" || true
