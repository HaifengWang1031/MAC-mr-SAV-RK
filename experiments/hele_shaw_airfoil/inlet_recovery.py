"""Re=10000 SAV rerun after the inlet energy correction, from zero initial data.

Serial T=1.1 pilot passes the former t=0.741 failure point before the separate
T=20 production run is allowed. The production outlet also uses the existing
backflow correction, which is zero on ordinary outflow. No scalar/velocity clip.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .wake_campaign import configurations, drive, prepare, verify_source


def recovery_configs() -> dict[str, dict[str, Any]]:
    base = {**configurations()['re10000'], 'inlet_treatment': 'energy_stable_pressure',
            'velocity_limit': 20., 'scalar_limit': .05, 'log_every': 10}
    return dict(pilot_inlet={**base, 'T': 1.1, 'snapshots': np.linspace(0., 1.1, 12).tolist()},
                re10000={**base, 'outlet_treatment': 'backflow_damped'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'start'])
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(cases=recovery_configs(), purpose=__doc__), flush=True)
        return
    if args.campaign is None:
        parser.error('--campaign is required')
    directory = args.campaign.resolve()
    verify_source(directory)
    if json.loads((directory/'release_checks.json').read_text())['status'] != 'passed':
        raise RuntimeError('Release/replay qualifications have not passed')
    drive(directory, ['pilot_inlet', 're10000'], render=True)


if __name__ == '__main__':
    main()
