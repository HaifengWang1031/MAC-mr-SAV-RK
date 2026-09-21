"""Fault injection worker, launched with a bounded timeout by pytest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from mpi4py import MPI
from experiments.mac_parallel import run as runner
from solver.mac_parallel import layout
from experiments import problems

comm = MPI.COMM_WORLD
scenario, scheme, destination = sys.argv[1:]
root = Path(destination)
config = {'experiment': 'decay', 'nx': 8, 'ny': 8, 'nu': .1,
          'T': .03, 'steps': [.01, .01, .01], 'scheme': scheme}
closed = []
original_close = runner.ParallelNS.close

def close(self):
    original_close(self)
    closed.append(True)

runner.ParallelNS.close = close
if scenario == 'kernel':
    original = layout.convection
    calls = 0
    def broken_kernel(*args):
        global calls
        calls += 1
        if comm.rank == 1 and calls == 3:
            raise ValueError('injected nonroot convection failure')
        return original(*args)
    layout.convection = broken_kernel
elif scenario == 'initial':
    def initial(self, amplitude=.1):
        def u(x, y):
            if comm.rank == 1:
                raise ValueError('injected nonroot initial field failure')
            return np.zeros_like(x)
        velocity = self.stokes.layout.vector(u, lambda x, y: np.zeros_like(x))
        return self.state(0., velocity)
    runner.ParallelNS.initial = initial
elif scenario in ('setup', 'force'):
    config['experiment'] = 'ns_mms'
    original_expressions = problems._expressions
    def expressions(*args):
        if scenario == 'setup' and comm.rank == 1:
            raise ValueError('injected nonroot setup failure')
        values = list(original_expressions(*args))
        original_force = values[7]
        def force(x, y, t):
            if comm.rank == 1 and t >= .01:
                raise ValueError('injected nonroot force failure')
            return original_force(x, y, t)
        values[7] = force
        return values
    problems._expressions = expressions
elif scenario == 'save':
    def save(*args):
        raise OSError('injected result saving failure')
    runner.save_result = save
else:
    raise ValueError(scenario)

path = runner.run_parallel(config, root=root, comm=comm)
assert closed == [True], (comm.rank, closed)
assert len(set(comm.allgather(str(path)))) == 1
if comm.rank == 0:
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['status'] == 'failed', manifest
    expected_steps = {'kernel': 1, 'force': 1, 'setup': 0, 'initial': 0, 'save': 3}[scenario]
    assert manifest['accepted_steps'] == expected_steps, manifest
    assert manifest['failures'][0]['rank'] == (0 if scenario == 'save' else 1), manifest
    assert 'injected' in manifest['error'], manifest
    if scenario in ('setup', 'save', 'initial'):
        assert not (path/'results.h5').exists()
        assert 'results_sha256' not in manifest
    print(json.dumps({'scenario': scenario, 'scheme': scheme, 'accepted_steps': expected_steps}), flush=True)

# Remove fault injection and independently reproduce the saved accepted prefix.
if scenario == 'kernel':layout.convection = original
if scenario in ('setup', 'force'):problems._expressions = original_expressions
if scenario in ('kernel', 'force'):
    reference_config = {**config, 'T': .01, 'steps': [.01]}
    reference = runner.run_parallel(reference_config, root=root, comm=comm)
    if comm.rank == 0:
        import h5py
        with h5py.File(path/'results.h5') as got, h5py.File(reference/'results.h5') as expected:
            for key in ('final/u', 'final/v'):
                np.testing.assert_allclose(got[key][...], expected[key][...], rtol=1e-12, atol=1e-13)
