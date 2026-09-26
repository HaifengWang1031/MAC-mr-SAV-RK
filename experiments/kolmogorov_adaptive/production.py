"""Run the full adaptive, fixed-control, and separate-analysis campaign."""
import argparse
import json
from pathlib import Path

from experiments.kolmogorov_adaptive.analyze_separate import analyze_separate
from experiments.kolmogorov_adaptive.run import run_campaign
from experiments.kolmogorov_adaptive.run_fixed_controls import run_fixed_controls
from experiments.workflow import PROJECT


def run_production(config_path: Path, *, root: Path = PROJECT) -> Path:
    config = json.loads(config_path.read_text())
    adaptive_batch = run_campaign(config, root=root)
    print(f'ADAPTIVE_BATCH={adaptive_batch}', flush=True)
    fixed_batch = run_fixed_controls(adaptive_batch, root=root)
    print(f'FIXED_CONTROL_BATCH={fixed_batch}', flush=True)
    report = analyze_separate(fixed_batch, root=root)
    print(f'REPORT={report}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    run_production(args.config, root=args.root)
