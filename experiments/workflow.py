"""Reproducible run/reuse/rerun and read-only analysis entry points."""
from pathlib import Path
from time import perf_counter
import json
import traceback
import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.operators import MACOperators
from solver.mac.stokes import DirectStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.registry import SCHEMES, make_scheme, scheme_info
from solver.integrate import Result, integrate, step_sizes
from .records import PROJECT, digest, begin_run, finish_run, load_record, provenance, run_identity, save_result, verified, write_json
from .forced_ns_convergence.isotropic import TARGETS
from .forced_rotation.model import KINDS

DEFAULTS={'nx':32,'ny':32,'lx':1.,'ly':1.,'nu':.1,'amplitude':.1,
          'T':.1,'dt':.001,'scheme':'sdirk2_mrsav','gamma':1.,'cache_size':4,'stokes_tolerance':1e-9,'snapshots':[],'snapshot_every':None,'log_every':1000,
          'm':1,'force_profile':'sin_2pi_m_x','forcing_kind':'A','k_f':5.,
          'ic_kind':'zero','ic_k_lo':4.,'ic_k_hi':12.,'ic_alpha':5./3.,'ic_seed':0,
          'ic_target':'max','ic_value':1.,'ic_symmetric':False,'ic_project':True,
          'ic_lattice_cutoff':100,'ic_lattice_amplitude':4.}


def effective_config(config: dict) -> dict:
    unknown=set(config)-set(DEFAULTS)-{'experiment','steps','lid_speed'}
    if unknown: raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    cfg={**DEFAULTS,**config}
    if cfg.get('experiment') not in ('cavity','forced_ns','trig_ns','rotation_ns','manufactured_ns'): raise ValueError('Unknown experiment')
    if cfg['experiment']=='cavity':
        cfg['lid_speed']=config.get('lid_speed',1.)
        if not np.isfinite(cfg['lid_speed']) or cfg['lid_speed']<=0: raise ValueError('Positive finite lid_speed required')
        cfg['boundary']='moving_top_lid_stationary_other_walls'
    elif 'lid_speed' in config: raise ValueError('lid_speed is only supported for cavity')
    if cfg['experiment']=='forced_ns':
        if type(cfg['m']) is not int or cfg['m']<1: raise ValueError('Force wavenumber m must be a positive integer')
        if cfg['force_profile'] not in ('sin_2pi_m_x','sin_x','clamped_streamfunction'): raise ValueError('Unknown force_profile')
        if cfg['force_profile']=='clamped_streamfunction' and not np.isclose(
                [cfg['lx'],cfg['ly']],2.*np.pi,rtol=0.,atol=1e-12).all():
            raise ValueError('Clamped force requires the (0, 2*pi)^2 domain')
        if cfg['ic_kind'] not in ('zero','isotropic_beams','lattice_streamfunction'): raise ValueError('Unknown ic_kind')
        if cfg['ic_kind']=='isotropic_beams':
            if cfg['ic_target'] not in TARGETS: raise ValueError('Unknown ic_target')
            if not np.isfinite([cfg['ic_k_lo'],cfg['ic_k_hi'],cfg['ic_alpha'],cfg['ic_value']]).all() \
                    or not 0.<cfg['ic_k_lo']<cfg['ic_k_hi'] or cfg['ic_value']<=0 or cfg['ic_alpha']<0:
                raise ValueError('Invalid isotropic band, slope or amplitude')
            if cfg['ic_k_hi']>min(cfg['nx'],cfg['ny'])/8.:
                raise ValueError('Isotropic band exceeds the wavenumbers this grid can resolve')
            if not isinstance(cfg['ic_symmetric'],bool) or not isinstance(cfg['ic_project'],bool):
                raise ValueError('Invalid ic_symmetric or ic_project')
            initial=f'isotropic_beams(k_lo={cfg["ic_k_lo"]},k_hi={cfg["ic_k_hi"]},'
            initial+=f'alpha={cfg["ic_alpha"]},seed={cfg["ic_seed"]},target={cfg["ic_target"]},'
            initial+=f'value={cfg["ic_value"]},symmetric={cfg["ic_symmetric"]},project={cfg["ic_project"]})'
        elif cfg['ic_kind']=='lattice_streamfunction':
            if type(cfg['ic_lattice_cutoff']) is not int or cfg['ic_lattice_cutoff']<1:
                raise ValueError('Positive integer lattice cutoff required')
            if not np.isfinite(cfg['ic_lattice_amplitude']) or cfg['ic_lattice_amplitude']<=0:
                raise ValueError('Positive finite lattice amplitude required')
            if not np.isclose([cfg['lx'],cfg['ly']],2.*np.pi,rtol=0.,atol=1e-12).all():
                raise ValueError('Lattice streamfunction requires the (0, 2*pi)^2 domain')
            if min(cfg['nx'],cfg['ny'])<=2*(np.floor(np.sqrt(cfg['ic_lattice_cutoff']))+1):
                raise ValueError('Grid does not resolve the lattice frequencies')
            if cfg['ic_project'] is not False:
                raise ValueError('Lattice discrete curl requires ic_project=false')
            initial=f'lattice_streamfunction(cutoff={cfg["ic_lattice_cutoff"]},'
            initial+=f'amplitude={cfg["ic_lattice_amplitude"]},signed_integer_modes,discrete_curl,no_normalization)'
            for key in ('ic_k_lo','ic_k_hi','ic_alpha','ic_seed','ic_target','ic_value','ic_symmetric'):
                cfg.pop(key)
        else:
            initial='zero'
        if 'ic_seed' in cfg and (type(cfg['ic_seed']) is not int or cfg['ic_seed']<0): raise ValueError('Invalid ic_seed')
        if cfg['force_profile']=='clamped_streamfunction':
            force='amplitude*[(1-cos(x))*sin(y), -sin(x)*(1-cos(y))]'
        else:
            force='[0, amplitude*sin(x)]' if cfg['force_profile']=='sin_x' else '[0, amplitude*sin(2*pi*m*x)]'
        cfg.update(force=force,initial_condition=initial,boundary='homogeneous_no_slip')
    elif 'm' in config: raise ValueError('m is only supported for forced_ns')
    if cfg['experiment']=='rotation_ns':
        if cfg['forcing_kind'] not in KINDS: raise ValueError('Unknown forcing_kind')
        if not np.isfinite(cfg['k_f']) or cfg['k_f']<=0: raise ValueError('Invalid forcing wavenumber k_f')
        if cfg['forcing_kind']=='A':
            components=f'(sin({cfg["k_f"]}*y), -sin({cfg["k_f"]}*x))'
        else:
            components=f'(cos({cfg["k_f"]}*y), -cos({cfg["k_f"]}*x))'
        cfg.update(force=f'amplitude*{components}',initial_condition='zero',boundary='homogeneous_no_slip')
    elif 'k_f' in config or 'forcing_kind' in config: raise ValueError('k_f is only supported for rotation_ns')
    if cfg['experiment']=='manufactured_ns':
        cfg.update(force='discrete MMS from sin²(pi*x) sin²(2*pi*y)',
                   initial_condition='a(0)*discrete_curl(streamfunction)',boundary='homogeneous_no_slip')
        if cfg['lx']!=1. or cfg['ly']!=1. or cfg['amplitude']<=0:
            raise ValueError('Manufactured case requires the unit square and positive amplitude')
    if cfg['experiment']=='trig_ns':
        cfg.update(force='[amplitude*cos(x), 0]',initial_condition='weighted_trig_vorticity',boundary='homogeneous_no_slip')
    if not isinstance(cfg['log_every'],int) or cfg['log_every']<1: raise ValueError('Invalid log_every')
    if cfg['scheme'] not in SCHEMES: raise ValueError('Unknown scheme')
    if scheme_info(cfg['scheme']).order == 1:
        cfg['pressure_semantics'] = scheme_info(cfg['scheme']).pressure_semantics
    MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    for key in ('nu','T'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f'Invalid {key}')
    if not np.isfinite(cfg['amplitude']) or not np.isfinite(cfg['gamma']) or cfg['gamma']<0:
        raise ValueError('Invalid amplitude or gamma')
    if not isinstance(cfg['cache_size'],int) or cfg['cache_size']<1: raise ValueError('Invalid cache size')
    if not np.isfinite(cfg['stokes_tolerance']) or cfg['stokes_tolerance']<=0:
        raise ValueError('Invalid stokes_tolerance')
    if 'steps' in config:
        if 'dt' in config: raise ValueError('Use dt or steps, not both')
        cfg.pop('dt')
    if cfg['snapshot_every'] is not None:
        if config.get('snapshots'): raise ValueError('Use snapshots or snapshot_every, not both')
        if not np.isfinite(cfg['snapshot_every']) or cfg['snapshot_every']<=0:
            raise ValueError('Invalid snapshot_every')
        count=int(np.floor(cfg['T']/cfg['snapshot_every']))
        cfg['snapshots']=[cfg['snapshot_every']*(index+1) for index in range(count)]
    schedule=step_sizes(cfg['T'],dt=cfg.get('dt'),steps=cfg.get('steps'))
    if any(not np.isfinite(t) or t<0 or t>cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Invalid snapshot times')
    # Record the actual sequence, including a possible shortened final step.
    cfg['actual_steps']=schedule.tolist()
    return cfg


def run_experiment(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg=effective_config(config)
    source=provenance()
    identity=run_identity(cfg,source)
    parent=Path(root)/'runs'/cfg['experiment']; parent.mkdir(parents=True,exist_ok=True)
    if not rerun:
        for candidate in sorted(parent.glob('*/manifest.json')):
            try: old=json.loads(candidate.read_text())
            except (OSError,ValueError): continue
            if old.get('identity')==identity and verified(old,candidate.parent): return candidate.parent
    directory, manifest = begin_run(parent, cfg, source,
                                    metadata={'schema_version': 1, 'rerun': rerun}, started_key='created_utc')
    run_id = directory.name
    if cfg['experiment'] in ('forced_ns','rotation_ns','manufactured_ns'):print(f'run_directory={directory}',flush=True)
    if cfg['experiment']=='trig_ns':print(f'run_directory={directory}',flush=True)
    grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    # The direct solver's residual gate is a numerical parameter of the run, so it is part of
    # the effective config: a larger grid can need a looser gate for the same problem.
    backend=DirectStokes(MACOperators(grid),cache_size=cfg['cache_size'],tolerance=cfg['stokes_tolerance'])
    model=MACNavierStokes(grid,cfg['nu'],backend=backend)
    if cfg['experiment']=='cavity':
        velocity0=np.zeros(grid.size)
    elif cfg['experiment'] in ('forced_ns','rotation_ns'):
        if cfg['ic_kind']=='zero':
            velocity0=np.zeros(grid.size)
        elif cfg['ic_kind']=='lattice_streamfunction':
            from .no_slip_reference_convergence.lattice import initial_velocity
            velocity0=initial_velocity(grid,cutoff=cfg['ic_lattice_cutoff'],amplitude=cfg['ic_lattice_amplitude'])
        else:
            from .forced_ns_convergence.isotropic import build_isotropic
            velocity0=build_isotropic(grid,k_lo=cfg['ic_k_lo'],k_hi=cfg['ic_k_hi'],alpha=cfg['ic_alpha'],
                                      seed=cfg['ic_seed'],target=cfg['ic_target'],value=cfg['ic_value'],
                                      symmetric=cfg['ic_symmetric'])
            if cfg['ic_project']:
                # One discrete Leray projection removes the O(h^2) sampling divergence of the
                # stream-function field, so the recorded initial state is discretely solenoidal.
                velocity0=model.backend.solve(velocity0,mass=1.,viscosity=0.).velocity
    elif cfg['experiment']=='manufactured_ns':
        from .manufactured_convergence.model import spatial_velocity, amplitude
        velocity0=amplitude(0.,cfg['amplitude'])*spatial_velocity(grid)
    elif cfg['experiment']=='trig_ns':
        from .forced_ns_convergence.model import initial_velocity_trig
        velocity0=initial_velocity_trig(grid)
    else:
        raise ValueError(f'No initial condition for {cfg["experiment"]}')
    initial=model.state(0.,velocity0)
    result=Result(initial,[0.],[model.diagnostics(initial)])
    metrics: dict={}
    start=perf_counter()
    with (directory/'run.log').open('w',buffering=1) as log:
        log.write(f'run_id={run_id}\nidentity={identity}\n')
        try:
            metrics['jit_warmup_seconds']=warmup()
            if cfg['experiment']=='cavity':
                from .cavity.model import lid_viscous_load
                boundary_load=lid_viscous_load(grid,cfg['nu'],cfg['lid_speed'])
                model.force=lambda t: boundary_load
            if cfg['experiment']=='forced_ns':
                if cfg['force_profile']=='clamped_streamfunction':
                    from .no_slip_reference_convergence.lattice import clamped_force_vector
                    constant_force=clamped_force_vector(grid,cfg['amplitude'])
                elif cfg['force_profile']=='sin_x':
                    from .no_slip_reference_convergence.lattice import force_vector as lattice_force
                    constant_force=lattice_force(grid,cfg['amplitude'])
                else:
                    from .forced_ns_convergence.model import force_vector
                    constant_force=force_vector(grid,cfg['amplitude'],cfg['m'])
                model.force=lambda t: constant_force
            if cfg['experiment']=='rotation_ns':
                from .forced_rotation.model import force_ab
                constant_force=force_ab(grid,cfg['forcing_kind'],cfg['amplitude'],cfg['k_f'])
                model.force=lambda t: constant_force
            if cfg['experiment']=='manufactured_ns':
                from .manufactured_convergence.model import configure
                configure(model,cfg['amplitude'])
            if cfg['experiment']=='trig_ns':
                from .forced_ns_convergence.model import force_vector_cos
                constant_force=force_vector_cos(grid,cfg['amplitude'])
                model.force=lambda t: constant_force
            scheme=make_scheme(cfg['scheme'],cfg['gamma'])
            def progress(step: int, time: float, elapsed: float) -> None:
                if step%cfg['log_every']==0 or step==len(cfg['actual_steps']):
                    log.write(f'accepted={step}/{len(cfg["actual_steps"])} t={time:.12g} elapsed={elapsed:.3f}s\n')
            result=integrate(model,scheme,initial,cfg['actual_steps'],snapshots=cfg['snapshots'],progress=progress)
            if cfg['experiment']=='cavity' and result.status=='complete':
                z=model.vector(result.final)
                steady_rhs=model.force(result.final.t)-model.nu*(model.ops.K@z)-model.nonlinear(z)
                projected=model.backend.solve(steady_rhs,mass=1.,viscosity=0.)
                metrics['steady_rhs_l2']=grid.norm(projected.velocity)
                metrics['reynolds']=cfg['lid_speed']*cfg['lx']/cfg['nu']
            metrics['integration_seconds']=result.seconds
            if isinstance(model.backend,DirectStokes):
                metrics.update(factorizations=model.backend.factorizations,
                               factorization_seconds=model.backend.factorization_seconds,
                               linear_solve_seconds=model.backend.solve_seconds)
        except Exception as error:
            result.status='failed'; result.error=f'{type(error).__name__}: {error}'
            log.write(traceback.format_exc())
        metrics['total_seconds']=perf_counter()-start
        info=scheme_info(cfg['scheme'])
        save_result(directory/'results.h5',result,metrics,info.n_stages,n_roots=info.n_roots,
                    pressure_semantics=cfg.get('pressure_semantics'))
        finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics,
                   finished_key='finished_utc', fields={'accepted_steps': len(result.times)-1,
                                                       'final_time': result.final.t})
        log.write(json.dumps({'status':result.status,'error':result.error,'metrics':metrics})+'\n')
    return directory
