"""Benchmark matrix-free TensorStokes in ten-step Kolmogorov integrations.

Purpose: determine whether the isolated D/D.T stencil speedup survives complete
SDIRK and mrSAV stepping for fixed and random positive step schedules.  The grid,
initial condition, schemes, schedules and solver tolerances remain fixed; only
the TensorStokes operator path changes.  CPU/wall time includes backend setup but
excludes the separately reported Numba warmup.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.kolmogorov_adaptive.benchmark_stokes_backends import (
    CONFIG as BASE_CONFIG,
    SCHEMES,
    make_scheme,
    schedules,
)
from experiments.kolmogorov_adaptive.model import force, initial_velocity
from experiments.workflow import PROJECT, provenance, write_json
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes


CONFIG = {
    **BASE_CONFIG,
    "repeats": 5,
    "primary_timing": "process_time_seconds_including_backend_setup_excluding_jit",
}


def run_case(
    config: dict[str, Any],
    grid: MACGrid,
    initial: np.ndarray,
    load: np.ndarray,
    scheme_name: str,
    schedule_name: str,
    steps: np.ndarray,
    matrix_free: bool,
    repeat: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model = MACNavierStokes(grid, config["nu"], cache_size=config["cache_size"])
    model.force = lambda _: load
    setup_cpu_start = process_time()
    setup_wall_start = perf_counter()
    backend = TensorStokes(
        model.ops,
        tolerance=config["stokes_tolerance"],
        krylov_tolerance=config["krylov_tolerance"],
        max_iterations=config["max_iterations"],
        matrix_free_operators=matrix_free,
        reuse_transform_buffers=matrix_free,
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
    mode = "matrix_free" if matrix_free else "sparse"
    if result.status != "complete":
        raise RuntimeError(
            f"{mode}/{scheme_name}/{schedule_name} failed: {result.error}"
        )
    return (
        {
            "repeat": repeat,
            "mode": mode,
            "scheme": scheme_name,
            "schedule": schedule_name,
            "setup_cpu_seconds": setup_cpu,
            "setup_wall_seconds": setup_wall,
            "integration_cpu_seconds": integration_cpu,
            "integration_wall_seconds": integration_wall,
            "total_cpu_seconds": setup_cpu + integration_cpu,
            "total_wall_seconds": setup_wall + integration_wall,
            "max_divergence": max(
                item["divergence_inf"] for item in result.diagnostics
            ),
            "final_kinetic": result.diagnostics[-1]["kinetic"],
            "stokes_solve_seconds": backend.solve_seconds,
            "stokes_solves": backend.solves,
            "schur_right_hand_sides": backend.right_hand_sides,
            "schur_iterations": backend.total_iterations,
            "schur_iterations_per_rhs": (
                backend.total_iterations / backend.right_hand_sides
            ),
        },
        model.vector(result.final),
    )


def median(records: list[dict[str, Any]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def summarize(
    records: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for scheme_name in SCHEMES:
        for schedule_name in ("fixed", "random"):
            groups = {
                mode: [
                    record
                    for record in records
                    if record["scheme"] == scheme_name
                    and record["schedule"] == schedule_name
                    and record["mode"] == mode
                ]
                for mode in ("sparse", "matrix_free")
            }
            comparison_group = [
                item
                for item in comparisons
                if item["scheme"] == scheme_name
                and item["schedule"] == schedule_name
            ]
            summary.append(
                {
                    "scheme": scheme_name,
                    "schedule": schedule_name,
                    "sparse_total_cpu_seconds_median": median(
                        groups["sparse"], "total_cpu_seconds"
                    ),
                    "matrix_free_total_cpu_seconds_median": median(
                        groups["matrix_free"], "total_cpu_seconds"
                    ),
                    "cpu_speedup_median": float(
                        np.median(
                            [item["sparse_over_matrix_free_cpu"] for item in comparison_group]
                        )
                    ),
                    "wall_speedup_median": float(
                        np.median(
                            [
                                item["sparse_over_matrix_free_wall"]
                                for item in comparison_group
                            ]
                        )
                    ),
                    "iteration_counts_equal": all(
                        item["iteration_counts_equal"] for item in comparison_group
                    ),
                    "max_final_velocity_relative_error": max(
                        item["final_velocity_relative_error"]
                        for item in comparison_group
                    ),
                    "max_matrix_free_divergence": max(
                        record["max_divergence"] for record in groups["matrix_free"]
                    ),
                }
            )
    return summary


def run_benchmark(config: dict[str, Any], output: Path) -> Path:
    grid = MACGrid(config["nx"], config["ny"], config["lx"], config["ly"])
    initial = initial_velocity(grid, config["epsilon"], config["initial_modes"])
    load = force(grid, config["forcing_m"])
    cases = schedules(config)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    jit_warmup_seconds = warmup()
    for repeat in range(1, config["repeats"] + 1):
        modes = (False, True) if repeat % 2 else (True, False)
        for scheme_name in SCHEMES:
            for schedule_name, steps in cases.items():
                paired_records: dict[bool, dict[str, Any]] = {}
                paired_velocities: dict[bool, np.ndarray] = {}
                for matrix_free in modes:
                    record, velocity = run_case(
                        config,
                        grid,
                        initial,
                        load,
                        scheme_name,
                        schedule_name,
                        steps,
                        matrix_free,
                        repeat,
                    )
                    records.append(record)
                    paired_records[matrix_free] = record
                    paired_velocities[matrix_free] = velocity
                    print(
                        f"repeat={repeat} scheme={scheme_name} "
                        f"schedule={schedule_name} mode={record['mode']} "
                        f"cpu={record['total_cpu_seconds']:.6f}s",
                        flush=True,
                    )
                sparse_record = paired_records[False]
                matrix_free_record = paired_records[True]
                denominator = np.linalg.norm(paired_velocities[False])
                comparisons.append(
                    {
                        "repeat": repeat,
                        "scheme": scheme_name,
                        "schedule": schedule_name,
                        "sparse_over_matrix_free_cpu": (
                            sparse_record["total_cpu_seconds"]
                            / matrix_free_record["total_cpu_seconds"]
                        ),
                        "sparse_over_matrix_free_wall": (
                            sparse_record["total_wall_seconds"]
                            / matrix_free_record["total_wall_seconds"]
                        ),
                        "iteration_counts_equal": (
                            sparse_record["schur_iterations"]
                            == matrix_free_record["schur_iterations"]
                        ),
                        "final_velocity_relative_error": float(
                            np.linalg.norm(
                                paired_velocities[True] - paired_velocities[False]
                            )
                            / denominator
                        ),
                    }
                )
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance(),
        "config": config,
        "fixed_steps": cases["fixed"].tolist(),
        "random_steps": cases["random"].tolist(),
        "jit_warmup_seconds": jit_warmup_seconds,
        "records": records,
        "comparisons": comparisons,
        "summary": summarize(records, comparisons),
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
        identity + "-matrix-free-10-step"
    ) / "timings.json"
    completed = run_benchmark(config, output)
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
