"""Benchmark shift reuse for ten fixed or random Kolmogorov-flow steps.

Purpose: measure whether ``TensorStokes`` removes the changing-step factorization
penalty seen in ``DirectStokes``.  Controls are the grid, initial condition, total
physical time, schemes and random schedule.  Recorded observables are end-to-end
CPU/wall time, backend setup/solve work, divergence, and the final-state difference
from the direct backend.  Improvement means lower total time while retaining the
direct solution to the configured Stokes tolerance.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.kolmogorov_adaptive.model import force, initial_velocity
from experiments.workflow import PROJECT, provenance, write_json
from solver.core import Scheme
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


CONFIG = {
    "nx": 128,
    "ny": 128,
    "lx": 2 * np.pi,
    "ly": 2 * np.pi,
    "nu": 0.02,
    "forcing_m": 4,
    "epsilon": 4.0,
    "initial_modes": 10,
    "gamma": 1000.0,
    "steps": 10,
    "fixed_step": 0.0025,
    "random_seed": 20260927,
    "random_multiplier_interval": [0.5, 1.5],
    "repeats": 3,
    "cache_size": 4,
    "stokes_tolerance": 1e-9,
    "krylov_tolerance": 1e-12,
    "max_iterations": 100,
    "primary_timing": "process_time_seconds_including_backend_setup",
}

SCHEMES = {
    "sdirk2": SDIRK2,
    "sdirk2_mrsav": SDIRK2MRSAV,
    "sdirk3": SDIRK3,
    "sdirk3_mrsav": SDIRK3MRSAV,
}


def make_scheme(name: str, gamma: float) -> Scheme:
    constructor = SCHEMES[name]
    return constructor(gamma) if name.endswith("mrsav") else constructor()


def schedules(config: dict[str, Any]) -> dict[str, np.ndarray]:
    fixed = np.full(config["steps"], config["fixed_step"])
    rng = np.random.default_rng(config["random_seed"])
    low, high = config["random_multiplier_interval"]
    random = rng.uniform(low, high, size=config["steps"])
    random *= fixed.sum() / random.sum()
    return {"fixed": fixed, "random": random}


def run_case(
    config: dict[str, Any],
    grid: MACGrid,
    initial: np.ndarray,
    load: np.ndarray,
    scheme_name: str,
    schedule_name: str,
    steps: np.ndarray,
    backend_name: str,
    repeat: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model = MACNavierStokes(grid, config["nu"], cache_size=config["cache_size"])
    model.force = lambda _: load
    setup_cpu_start = process_time()
    setup_wall_start = perf_counter()
    backend: DirectStokes | TensorStokes
    if backend_name == "direct":
        backend = DirectStokes(
            model.ops,
            cache_size=config["cache_size"],
            tolerance=config["stokes_tolerance"],
        )
    else:
        backend = TensorStokes(
            model.ops,
            tolerance=config["stokes_tolerance"],
            krylov_tolerance=config["krylov_tolerance"],
            max_iterations=config["max_iterations"],
        )
    setup_cpu = process_time() - setup_cpu_start
    setup_wall = perf_counter() - setup_wall_start
    model.backend = backend
    state = model.state(0.0, initial.copy())
    integration_cpu_start = process_time()
    integration_wall_start = perf_counter()
    result = integrate(model, make_scheme(scheme_name, config["gamma"]), state, steps)
    integration_cpu = process_time() - integration_cpu_start
    integration_wall = perf_counter() - integration_wall_start
    if result.status != "complete":
        raise RuntimeError(
            f"{backend_name}/{scheme_name}/{schedule_name} failed: {result.error}"
        )
    record: dict[str, Any] = {
        "repeat": repeat,
        "backend": backend_name,
        "scheme": scheme_name,
        "schedule": schedule_name,
        "setup_cpu_seconds": setup_cpu,
        "setup_wall_seconds": setup_wall,
        "integration_cpu_seconds": integration_cpu,
        "integration_wall_seconds": integration_wall,
        "total_cpu_seconds": setup_cpu + integration_cpu,
        "total_wall_seconds": setup_wall + integration_wall,
        "max_divergence": max(item["divergence_inf"] for item in result.diagnostics),
        "final_kinetic": result.diagnostics[-1]["kinetic"],
    }
    if isinstance(backend, DirectStokes):
        record.update(
            factorizations=backend.factorizations,
            factorization_seconds=backend.factorization_seconds,
            linear_solve_seconds=backend.solve_seconds,
        )
    else:
        record.update(
            spectral_setup_seconds=backend.setup_seconds,
            stokes_solve_seconds=backend.solve_seconds,
            stokes_solves=backend.solves,
            schur_right_hand_sides=backend.right_hand_sides,
            schur_iterations=backend.total_iterations,
            schur_iterations_per_rhs=backend.total_iterations / backend.right_hand_sides,
        )
    return record, model.vector(result.final)


def median(records: list[dict[str, Any]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for scheme_name in SCHEMES:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for backend_name in ("direct", "tensor"):
            for schedule_name in ("fixed", "random"):
                group = [
                    record
                    for record in records
                    if record["scheme"] == scheme_name
                    and record["backend"] == backend_name
                    and record["schedule"] == schedule_name
                ]
                grouped[backend_name, schedule_name] = group
                summary = {
                    "scheme": scheme_name,
                    "backend": backend_name,
                    "schedule": schedule_name,
                    "total_cpu_seconds_median": median(group, "total_cpu_seconds"),
                    "total_wall_seconds_median": median(group, "total_wall_seconds"),
                    "setup_cpu_seconds_median": median(group, "setup_cpu_seconds"),
                    "integration_cpu_seconds_median": median(group, "integration_cpu_seconds"),
                    "max_divergence": max(record["max_divergence"] for record in group),
                }
                if backend_name == "direct":
                    summary["factorizations"] = sorted(
                        {record["factorizations"] for record in group}
                    )
                else:
                    summary["schur_iterations_per_rhs_median"] = median(
                        group, "schur_iterations_per_rhs"
                    )
                summaries.append(summary)
        for schedule_name in ("fixed", "random"):
            direct_time = median(grouped["direct", schedule_name], "total_cpu_seconds")
            tensor_time = median(grouped["tensor", schedule_name], "total_cpu_seconds")
            comparisons.append(
                {
                    "scheme": scheme_name,
                    "schedule": schedule_name,
                    "direct_over_tensor_speedup": direct_time / tensor_time,
                    "tensor_final_velocity_relative_error_max": max(
                        record["final_velocity_relative_error"]
                        for record in grouped["tensor", schedule_name]
                    ),
                }
            )
        for backend_name in ("direct", "tensor"):
            fixed_time = median(grouped[backend_name, "fixed"], "total_cpu_seconds")
            random_time = median(grouped[backend_name, "random"], "total_cpu_seconds")
            comparisons.append(
                {
                    "scheme": scheme_name,
                    "backend": backend_name,
                    "random_over_fixed_total_cpu": random_time / fixed_time,
                }
            )
    return summaries, comparisons


def run_benchmark(config: dict[str, Any], output: Path) -> Path:
    grid = MACGrid(config["nx"], config["ny"], config["lx"], config["ly"])
    initial = initial_velocity(grid, config["epsilon"], config["initial_modes"])
    load = force(grid, config["forcing_m"])
    cases = schedules(config)
    records: list[dict[str, Any]] = []
    jit_warmup_seconds = warmup()
    for repeat in range(1, config["repeats"] + 1):
        for scheme_name in SCHEMES:
            for schedule_name, steps in cases.items():
                reference: np.ndarray | None = None
                for backend_name in ("direct", "tensor"):
                    record, final = run_case(
                        config,
                        grid,
                        initial,
                        load,
                        scheme_name,
                        schedule_name,
                        steps,
                        backend_name,
                        repeat,
                    )
                    if reference is None:
                        reference = final
                        record["final_velocity_relative_error"] = 0.0
                    else:
                        record["final_velocity_relative_error"] = float(
                            np.linalg.norm(final - reference) / np.linalg.norm(reference)
                        )
                    records.append(record)
                    print(
                        f"repeat={repeat} scheme={scheme_name} schedule={schedule_name} "
                        f"backend={backend_name} total_cpu={record['total_cpu_seconds']:.6f}s",
                        flush=True,
                    )
    summary, comparisons = summarize(records)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance(),
        "config": config,
        "fixed_steps": cases["fixed"].tolist(),
        "random_steps": cases["random"].tolist(),
        "total_time": float(cases["fixed"].sum()),
        "jit_warmup_seconds": jit_warmup_seconds,
        "records": records,
        "summary": summary,
        "comparisons": comparisons,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"])
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive")
    config = {**CONFIG, "repeats": arguments.repeats}
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = arguments.output or PROJECT / "reports/timings" / (
        identity + "-tau-decoupled-10-step"
    ) / "timings.json"
    completed = run_benchmark(config, output)
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
