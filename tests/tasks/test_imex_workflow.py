"""First-order public run/read/analyze and failure-prefix storage contracts."""
import json
import h5py
import pytest

from experiments.analysis import analyze_runs
from experiments.workflow import load_record, run_experiment
from solver.schemes import IMEXEuler, IMEXEulerMRSAV
from solver.schemes.registry import DEFAULT_SCHEMES, make_scheme, scheme_info

NAMES = ('imex_euler', 'imex_euler_mrsav')


@pytest.mark.parametrize('name', NAMES)
def test_no_slip_public_run_single_stage_storage_reuse_and_analysis(tmp_path, name, monkeypatch):
    cfg = {'experiment': 'manufactured_ns', 'scheme': name, 'nx': 8, 'ny': 6,
           'T': .03, 'steps': [.01, .02], 'snapshots': [0., .03]}
    directory = run_experiment(cfg, root=tmp_path)
    stored, manifest = load_record(directory, require_complete=True)
    assert run_experiment(cfg, root=tmp_path) == directory
    assert stored['pressure_semantics'] == 'euler_step_multiplier_mean_zero'
    with h5py.File(directory/'results.h5') as data:
        assert data.attrs['pressure_semantics'] == stored['pressure_semantics']
        assert data['stages/residual'].shape == (2, 1)
        assert data['roots/candidates'].shape == (2, 1, 1)
        assert data['final/stage_pressure'].shape == (1, 6, 8)
        assert (data['roots/count'][:] == 0).all()
        assert (data['corrections/count'][:] == (1 if name.endswith('_mrsav') else 0)).all()
        assert abs(data['final/t'][()]-.03) < 1e-14
    import experiments.workflow as workflow
    monkeypatch.setattr(workflow, 'make_scheme', lambda *args: (_ for _ in ()).throw(AssertionError('analysis invoked solver')))
    report = analyze_runs([directory], root=tmp_path)
    assert json.loads((report/'analysis.json').read_text())['status'] == 'complete'
    assert len(list((tmp_path/'runs/manufactured_ns').iterdir())) == 1


@pytest.mark.parametrize('name,cls', list(zip(NAMES, (IMEXEuler, IMEXEulerMRSAV))))
def test_failed_one_stage_run_keeps_accepted_prefix(tmp_path, name, cls, monkeypatch):
    step = cls.step
    def fail_after_first(self, model, state, dt):
        if state.t > 0: raise RuntimeError('injected second-step failure')
        return step(self, model, state, dt)
    monkeypatch.setattr(cls, 'step', fail_after_first)
    cfg = {'experiment': 'forced_ns', 'scheme': name, 'nx': 8, 'ny': 6, 'T': .03, 'dt': .01}
    directory = run_experiment(cfg, root=tmp_path)
    _, manifest = load_record(directory)
    assert manifest['status'] == 'failed' and manifest['accepted_steps'] == 1
    with h5py.File(directory/'results.h5') as data:
        assert data['times'][:].tolist() == [0., .01]
        assert data['stages/residual'].shape == (1, 1)
        assert data['roots/candidates'].shape == (1, 1, 1)
    with pytest.raises(ValueError, match='Complete run'): load_record(directory, require_complete=True)
    assert run_experiment(cfg, root=tmp_path) != directory


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('module', ['channel_ns', 'periodic_ns'])
def test_euler_adaptive_request_is_rejected_before_computation(tmp_path, name, module):
    from importlib import import_module
    run = import_module('experiments.'+module+'.run').run
    with pytest.raises(ValueError, match='step-doubling'):
        run({'scheme': name, 'mode': 'adaptive'}, root=tmp_path)
    assert not (tmp_path/'runs').exists()


def test_scheme_metadata_and_legacy_default_studies():
    from experiments.channel_ns.validate import DEFAULTS as channel
    from experiments.channel_ns.benchmark import DEFAULTS as benchmark
    from experiments.periodic_ns.validate import DEFAULTS as periodic
    for defaults in (channel, periodic, benchmark):
        assert defaults['schemes'] == list(DEFAULT_SCHEMES)
    for name in NAMES:
        info = scheme_info(name)
        assert (info.order, info.n_stages, info.n_roots, info.estimator_order) == (1, 1, 1, None)
        assert make_scheme(name).name == name
    with pytest.raises(ValueError, match='Unknown scheme'): make_scheme('unknown')


def test_refinement_analysis_uses_saved_fields_and_rejects_corruption(tmp_path, monkeypatch):
    from experiments.imex_euler import validate as verification
    directory = verification.validate({'backends': ['legendre'], 'schemes': ['imex_euler'],
                                       'counts': [40, 80, 160], 'gammas': [1.]}, root=tmp_path)
    monkeypatch.setattr(verification, '_setup', lambda *args: (_ for _ in ()).throw(AssertionError('analysis constructed model')))
    report = verification.analyze_validation(directory, root=tmp_path)
    data = json.loads((report/'analysis.json').read_text())
    assert data['first_order_observed']
    member = directory/data['rows'][0]['members'][0]['artifact']
    with member.open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'): verification.analyze_validation(directory, root=tmp_path)


def test_periodic_euler_reader_checks_pressure_semantics_after_checksum_gate(tmp_path):
    from experiments.periodic_ns.run import load_periodic, run
    from experiments.workflow import digest, write_json
    directory = run({'scheme': 'imex_euler', 'nx': 18, 'ny': 15, 'T': .01}, root=tmp_path)
    with h5py.File(directory/'results.h5', 'a') as data:
        data.attrs['pressure_semantics'] = 'incremental_stage_multiplier_mean_zero'
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['results_sha256'] = digest(directory/'results.h5')
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='pressure semantics'): load_periodic(directory)
