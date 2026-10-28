"""Compute a short, accuracy-aware comparison of the existing spatial backends.

The independent reference is the smooth, no-slip manufactured Navier--Stokes
solution in experiments.problems. No solver implementation is changed here.
Initialization and integration are timed separately; reference evaluation and
microbenchmarks are outside those timers. Analyze an explicit saved run later.
"""
import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from experiments.problems import _expressions, initial_velocity
from experiments.workflow import PROJECT, digest, provenance, write_json
from solver.core import Scheme, State
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel

Array = NDArray[np.float64]
Model = MACNavierStokes | SpectralModel
CONFIG = Path(__file__).parent / "configs/preliminary.json"


def schedules(config: dict[str, Any]) -> dict[str, Array]:
    """Prescribed schedules with equal total time; these are not controller runs."""
    fixed = np.full(config["steps"], config["fixed_step"], dtype=float)
    rng = np.random.default_rng(config["random_seed"])
    low, high = config["random_multiplier_interval"]
    variable = rng.uniform(low, high, size=len(fixed))
    variable *= fixed.sum() / variable.sum()
    return {"fixed": fixed, "variable": variable}


def make_scheme(name: str, gamma: float) -> Scheme:
    if name == "sdirk3":
        return SDIRK3()
    if name == "sdirk3_mrsav":
        return SDIRK3MRSAV(gamma)
    raise ValueError(f"Unsupported scheme {name}")


def build_model(config: dict[str, Any], backend: str, size: int) -> tuple[Model, State]:
    """Set up the same continuum force, using each space's natural representation."""
    lx, ly, nu, amplitude = (config[key] for key in ("lx", "ly", "nu", "amplitude"))
    expressions = _expressions(lx, ly, nu)
    model: Model
    if backend == "legendre_lu":
        spectral = SpectralModel(
            Space(size, lx, ly), nu, tolerance=config["stokes_tolerance"],
            cache_size=config["cache_size"], dealias=config["dealias"],
        )

        def projected(index: int, scale: float) -> Array:
            return spectral.project(
                lambda x, y: scale * expressions[index](x, y, 0.),
                extra=max(config["projection_extra"], spectral.quadrature_extra),
            )

        velocity = np.concatenate([projected(j, amplitude) for j in (0, 1)])
        linear = np.concatenate([projected(j, amplitude) for j in (7, 9)])
        quadratic = np.concatenate([projected(j, amplitude**2) for j in (8, 10)])
        model = spectral
    elif backend in ("mac_direct", "mac_tensor"):
        grid = MACGrid(size, size, lx, ly)
        mac = MACNavierStokes(grid, nu, cache_size=config["cache_size"])
        if backend == "mac_direct":
            mac.backend = DirectStokes(mac.ops, cache_size=config["cache_size"],
                                       tolerance=config["stokes_tolerance"])
        else:
            mac.backend = TensorStokes(
                mac.ops, tolerance=config["stokes_tolerance"],
                krylov_tolerance=config["krylov_tolerance"],
                max_iterations=config["max_iterations"],
            )
        velocity = initial_velocity(grid, amplitude=amplitude)

        def sampled(indices: tuple[int, int], scale: float) -> Array:
            xu, yu = grid.coordinates("u")
            xv, yv = grid.coordinates("v")
            u = scale * expressions[indices[0]](xu, yu, 0.)
            v = scale * expressions[indices[1]](xv, yv, 0.)
            u[:, [0, -1]] = 0.
            v[[0, -1], :] = 0.
            return grid.pack(u, v)

        linear = sampled((7, 9), amplitude)
        quadratic = sampled((8, 10), amplitude**2)
        model = mac
    else:
        raise ValueError(f"Unknown backend {backend}")
    model.force = lambda t: np.exp(-t) * linear + np.exp(-2*t) * quadratic
    return model, model.state(0., velocity)


def mac_evaluate(grid: MACGrid, velocity: Array, x: Array, y: Array) -> tuple[Array, Array]:
    """Bilinear face-field reconstruction, with zero tangential wall values."""
    u, v = grid.unpack(velocity)
    ux = np.arange(grid.nx + 1) * grid.hx
    uy = np.r_[0., (np.arange(grid.ny) + .5)*grid.hy, grid.ly]
    vx = np.r_[0., (np.arange(grid.nx) + .5)*grid.hx, grid.lx]
    vy = np.arange(grid.ny + 1) * grid.hy
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([yy.ravel(), xx.ravel()])
    sampled_u = RegularGridInterpolator((uy, ux), np.pad(u, ((1, 1), (0, 0))))(points)
    sampled_v = RegularGridInterpolator((vy, vx), np.pad(v, ((0, 0), (1, 1))))(points)
    return sampled_u.reshape(yy.shape), sampled_v.reshape(yy.shape)


def relative_error(
    config: dict[str, Any], backend: str, size: int, velocity: Array, t: float, nodes: int,
) -> float:
    """Relative physical vector L2 error on a common Gauss quadrature."""
    z, weights = np.polynomial.legendre.leggauss(nodes)
    x, y = .5*config["lx"]*(z+1), .5*config["ly"]*(z+1)
    if backend == "legendre_lu":
        space = Space(size, config["lx"], config["ly"])
        u = space.evaluate(velocity[:size**2].reshape(size, size), x, y)
        v = space.evaluate(velocity[size**2:].reshape(size, size), x, y)
    else:
        u, v = mac_evaluate(MACGrid(size, size, config["lx"], config["ly"]), velocity, x, y)
    xx, yy = np.meshgrid(x, y)
    expressions = _expressions(config["lx"], config["ly"], config["nu"])
    exact_u, exact_v = [config["amplitude"]*expressions[j](xx, yy, t) for j in (0, 1)]
    quadrature = np.outer(weights, weights)
    error_squared = np.sum(quadrature * ((u-exact_u)**2 + (v-exact_v)**2))
    norm_squared = np.sum(quadrature * (exact_u**2 + exact_v**2))
    return float(np.sqrt(error_squared/norm_squared))


def micro_cpu(operation: Callable[[], Any], repeats: int) -> float:
    """Warm-call CPU median, reported separately from integration time."""
    operation()
    samples = []
    for _ in range(repeats):
        start = process_time()
        operation()
        samples.append(process_time()-start)
    return float(np.median(samples))


def run_case(
    config: dict[str, Any], backend: str, size: int, scheme_name: str,
    schedule_name: str, steps: Array, repeat: int,
) -> tuple[dict[str, Any], Array]:
    cpu, wall = process_time(), perf_counter()
    model, initial = build_model(config, backend, size)
    setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
    scheme = make_scheme(scheme_name, config["gamma"])
    cpu, wall = process_time(), perf_counter()
    # integrate's historical annotation names MAC; the shared model seam is structural.
    result = integrate(cast(MACNavierStokes, model), scheme, initial, steps)
    integration_cpu, integration_wall = process_time()-cpu, perf_counter()-wall
    if result.status != "complete" or len(result.times) != len(steps)+1:
        raise RuntimeError(f"{backend}/{size}/{scheme_name}: {result.error}")
    final = model.vector(result.final)
    record: dict[str, Any] = {
        "backend": backend, "size": size, "scheme": scheme_name,
        "schedule": schedule_name, "repeat": repeat, "steps": len(steps),
        "final_time": result.final.t, "final_r": result.final.r,
        "velocity_dofs": len(final), "pressure_dofs": size**2-1,
        "setup_cpu_seconds": setup_cpu, "setup_wall_seconds": setup_wall,
        "integration_cpu_seconds": integration_cpu, "integration_wall_seconds": integration_wall,
        "total_cpu_seconds": setup_cpu+integration_cpu,
        "total_wall_seconds": setup_wall+integration_wall,
        "max_physical_divergence": max(d["divergence_inf"] for d in result.diagnostics),
        "final_constraint_inf": float(np.max(np.abs(model.apply_D(final)))),
        "max_stage_residual": max(s["residual"] for stages in result.stages for s in stages),
        "max_stage_continuity_residual": max(s["continuity_residual"] for stages in result.stages for s in stages),
    }
    if isinstance(model, SpectralModel):
        record.update(factorizations=model.stokes.factorizations,
                      factorization_wall_seconds=model.stokes.factorization_seconds,
                      nonlinear_quadrature_nodes=size+model.quadrature_extra)
    elif isinstance(model.backend, DirectStokes):
        record.update(factorizations=model.backend.factorizations,
                      factorization_wall_seconds=model.backend.factorization_seconds,
                      linear_solve_wall_seconds=model.backend.solve_seconds)
    elif isinstance(model.backend, TensorStokes):
        record.update(factorizations=0, schur_iterations=model.backend.total_iterations,
                      schur_rhs=model.backend.right_hand_sides,
                      linear_solve_wall_seconds=model.backend.solve_seconds)
    record["relative_L2"] = relative_error(config, backend, size, final, result.final.t,
                                            config["error_quadrature"])
    record["relative_L2_refined"] = relative_error(config, backend, size, final, result.final.t,
                                                    config["error_quadrature_refined"])
    record["quadrature_relative_difference"] = abs(record["relative_L2"]-record["relative_L2_refined"]) \
        / max(record["relative_L2_refined"], np.finfo(float).tiny)
    record["quadrature_passed"] = record["quadrature_relative_difference"] \
        <= config["error_quadrature_relative_gate"]
    count = config["micro_repeats"]
    record["nonlinear_cpu_seconds"] = micro_cpu(lambda: model.nonlinear(final), count)
    record["diagnostics_cpu_seconds"] = micro_cpu(lambda: model.diagnostics(result.final), count)
    record["cached_stokes_cpu_seconds"] = micro_cpu(
        lambda: model.solve(final, mass=1., viscosity=config["nu"]*.25*float(steps[-1])), count,
    )
    return record, final


def run_benchmark(config: dict[str, Any], *, root: Path = PROJECT) -> Path:
    """Persist all repeats, schedules, final fields, checksums and source provenance."""
    for key in ("lx", "ly", "nu", "amplitude", "fixed_step"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if min(config["steps"], config["repeats"], config["micro_repeats"]) < 1:
        raise ValueError("steps and repeat counts must be positive")
    for name in config["schemes"]:
        make_scheme(name, config["gamma"])
    sequence = schedules(config)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8]
    directory = root / "runs/legendre_mac_benchmark" / identifier
    directory.mkdir(parents=True)
    write_json(directory/"config.json", config)
    write_json(directory/"schedules.json", {name: steps.tolist() for name, steps in sequence.items()})
    manifest: dict[str, Any] = {
        "id": identifier, "status": "running", "purpose": "existing_legendre_vs_mac_time_and_accuracy",
        "provenance": provenance(), "completed_cases": 0, "artifacts": {},
    }
    write_json(directory/"manifest.json", manifest)
    # Imports, symbolic manufacture and Numba compilation are excluded from timers.
    _expressions(config["lx"], config["ly"], config["nu"])
    manifest["excluded_mac_jit_warmup_wall_seconds"] = warmup()
    cases = [(backend, size) for backend in ("mac_direct", "mac_tensor") for size in config["mac_grids"]]
    cases += [("legendre_lu", size) for size in config["legendre_modes"]]
    records: list[dict[str, Any]] = []
    fields: dict[str, Array] = {}
    manifest["planned_cases"] = len(cases)*len(config["schemes"])*len(sequence)*config["repeats"]
    try:
        with (directory/"run.log").open("w") as log:
            for repeat in range(config["repeats"]):
                ordered = cases if repeat % 2 == 0 else list(reversed(cases))
                for backend, size in ordered:
                    for scheme in config["schemes"]:
                        for name, steps in sequence.items():
                            record, final = run_case(config, backend, size, scheme, name, steps, repeat)
                            key = f"{backend}_{size}_{scheme}_{name}_{repeat}"
                            record["field_key"] = key
                            fields[key] = final
                            records.append(record)
                            write_json(directory/"results.json", {"records": records})
                            manifest["completed_cases"] = len(records)
                            write_json(directory/"manifest.json", manifest)
                            message = f"{len(records)}/{manifest['planned_cases']} {key}: " \
                                f"CPU={record['total_cpu_seconds']:.6f}s error={record['relative_L2_refined']:.3e}"
                            print(message, flush=True)
                            log.write(message+"\n")
                            log.flush()
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        np.savez_compressed(directory/"final_fields.npz", **cast(Any, fields))
        for name in ("config.json", "schedules.json", "results.json", "final_fields.npz", "run.log"):
            if (directory/name).exists():
                manifest["artifacts"][name] = digest(directory/name)
        write_json(directory/"manifest.json", manifest)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(f"RUN={run_benchmark(json.loads(args.config.read_text()))}")


if __name__ == "__main__":
    main()
