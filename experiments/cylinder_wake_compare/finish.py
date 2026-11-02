"""Program-owned completion: wait for owned runs, calculate checks, then render saved data."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from experiments.workflow import PROJECT, digest, write_json
from .checks import verify
from .analyze import analyze


def _validate_source(directory: Path, manifest: dict) -> None:
    """Check immutable worker sources and the numerical implementation used by checks."""
    source = json.loads((directory/'source.json').read_text())
    if digest(directory/'source.json') != manifest['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    for name, checksum in source['files'].items():
        if digest(directory/'source'/name) != checksum:
            raise ValueError('Frozen source file changed: '+name)
        if name.startswith(('solver/', 'experiments/cylinder_ns/', 'experiments/fem_ns/')) \
                and digest(PROJECT/name) != checksum:
            raise ValueError('Numerical source changed before sensitivity checks: '+name)


def finish(directory: Path) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    try:
        _validate_source(directory, manifest)
        while True:
            states = []
            for case in manifest['cases']:
                output = Path(case['directory'])/'result.json'
                if output.exists():
                    state = json.loads(output.read_text())
                    states.append(state['status']); case['status'] = state['status']
                else:
                    try:
                        os.kill(case['pid'], 0)
                        states.append('running')
                    except ProcessLookupError:
                        if output.exists():
                            state = json.loads(output.read_text())
                            states.append(state['status']); case['status'] = state['status']
                        else:
                            raise RuntimeError('A cylinder worker exited without a final record')
            write_json(directory/'manifest.json', manifest)
            if all(state != 'running' for state in states): break
            time.sleep(10)
        if any(state != 'complete' for state in states): raise RuntimeError('A cylinder worker failed; inspect its log and preserved run')
        manifest['status'] = 'computed'
        write_json(directory/'manifest.json', manifest)
        _validate_source(directory, manifest)
        verify(directory)
        manifest.update(status='complete', checks_sha256=digest(directory/'checks.json'),
                        finished=datetime.now(timezone.utc).isoformat())
        write_json(directory/'manifest.json', manifest)
        report = analyze(directory)
        write_json(directory/'delivery.json', {'status': 'complete', 'report': str(report),
                                              'analysis_sha256': digest(report/'analysis.json')})
        return report
    except Exception as error:
        write_json(directory/'delivery.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(finish(args.run), flush=True)


if __name__ == '__main__': main()
