"""MPI fault tests must time out rather than hang the whole test suite."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import pytest

pytest.importorskip('petsc4py')

@pytest.mark.parametrize('ranks', [2, 4])
@pytest.mark.parametrize('scenario,scheme', [('kernel','sdirk2'), ('kernel','sdirk2_mrsav'),
                                            ('force','sdirk2_mrsav'), ('setup','sdirk2'), ('initial','sdirk2'), ('save','sdirk2')])
def test_parallel_failure_exits_and_records_prefix(tmp_path, ranks, scenario, scheme):
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('MPI launcher unavailable')
    worker = Path(__file__).with_name('mpi_failure_worker.py')
    process = subprocess.Popen([launcher, '-n', str(ranks), sys.executable, str(worker),
                                scenario, scheme, str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f'MPI failure recovery hung: {output}')
    assert process.returncode == 0, output
