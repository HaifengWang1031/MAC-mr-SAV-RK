"""Compare sparse and matrix-free D/D.T paths in TensorStokes.

Purpose: measure whether serial Numba MAC stencils reduce complete TensorStokes
CPU and wall time while retaining the same scalar Schur CG algorithm.  Grid,
right-hand sides, viscosity sequence, FFT worker count and tolerances remain
fixed.  JIT dispatch is warmed and reported separately from timed solves.
Acceptance requires identical iteration traces, existing algebraic checks, and
an end-to-end speedup rather than an isolated stencil microbenchmark gain.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.fft import set_workers

from experiments.workflow import PROJECT, provenance, write_json
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.operators import MACOperators
from solver.mac.tensor_stokes import TensorStokes


CONFIG = {
    "grids": [128, 512],
    "solves": {"128": 20, "512": 5},
    "right_hand_sides": [1, 2],
    "repeats": 5,
    "seed": 20260927,
    "lx": 2 * np.pi,
    "ly": 2 * np.pi,
    "mass": 1.0,
    "viscosity_interval": [8e-6, 2e-5],
    "fft_workers": 1,
    "stokes_tolerance": 1e-9,
    "krylov_tolerance": 1e-12,
    "max_iterations": 100,
    "primary_timing": "process_time_seconds_excluding_backend_setup_and_jit",
}


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def run_solver(
    config: dict[str, Any],
    operators: MACOperators,
    right_hand_sides: list[np.ndarray],
    viscosities: np.ndarray,
    matrix_free: bool,
    repeat: int,
    reuse_transform_buffers: bool = False,
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    setup_cpu_start = process_time()
    setup_wall_start = perf_counter()
    solver = TensorStokes(
        operators,
        tolerance=config["stokes_tolerance"],
        krylov_tolerance=config["krylov_tolerance"],
        max_iterations=config["max_iterations"],
        matrix_free_operators=matrix_free,
        reuse_transform_buffers=reuse_transform_buffers,
    )
    setup_cpu = process_time() - setup_cpu_start
    setup_wall = perf_counter() - setup_wall_start

    solutions: list[tuple[np.ndarray, np.ndarray]] = []
    iteration_trace: list[list[int]] = []
    residual = 0.0
    divergence = 0.0
    cpu_start = process_time()
    wall_start = perf_counter()
    for rhs, viscosity in zip(right_hand_sides, viscosities, strict=True):
        result = solver.solve(rhs, mass=config["mass"], viscosity=float(viscosity))
        solutions.append((result.velocity.copy(), result.pressure.copy()))
        iteration_trace.append(list(solver.last_iterations))
        residual = max(residual, result.residual)
        divergence = max(divergence, result.divergence_inf)
    wall_seconds = perf_counter() - wall_start
    cpu_seconds = process_time() - cpu_start
    return (
        {
            "repeat": repeat,
            "mode": (
                "matrix_free_reuse"
                if reuse_transform_buffers
                else "matrix_free" if matrix_free else "sparse"
            ),
            "setup_cpu_seconds": setup_cpu,
            "setup_wall_seconds": setup_wall,
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "schur_iterations_per_rhs": (
                solver.total_iterations / solver.right_hand_sides
            ),
            "iteration_trace": iteration_trace,
            "max_residual": residual,
            "max_divergence": divergence,
        },
        solutions,
    )


def run_grid(
    config: dict[str, Any], size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid = MACGrid(size, size, config["lx"], config["ly"])
    operators = MACOperators(grid)
    count = int(config["solves"][str(size)])
    lower, upper = config["viscosity_interval"]
    viscosities = np.linspace(float(lower), float(upper), count)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for columns in config["right_hand_sides"]:
        rng = np.random.default_rng(config["seed"] + 1000 * size + columns)
        right_hand_sides = [
            rng.normal(size=(grid.size, columns))
            if columns > 1
            else rng.normal(size=grid.size)
            for _ in range(count)
        ]
        for matrix_free in (False, True):
            warm = TensorStokes(
                operators,
                matrix_free_operators=matrix_free,
                reuse_transform_buffers=matrix_free,
            )
            warm.solve(
                right_hand_sides[0],
                mass=config["mass"],
                viscosity=float(viscosities[0]),
            )

        for repeat in range(1, config["repeats"] + 1):
            modes = (False, True) if repeat % 2 else (True, False)
            paired_records: dict[bool, dict[str, Any]] = {}
            paired_solutions: dict[bool, list[tuple[np.ndarray, np.ndarray]]] = {}
            for matrix_free in modes:
                record, solutions = run_solver(
                    config,
                    operators,
                    right_hand_sides,
                    viscosities,
                    matrix_free,
                    repeat,
                    reuse_transform_buffers=matrix_free,
                )
                record.update(grid=size, right_hand_sides=columns, solves=count)
                records.append(record)
                paired_records[matrix_free] = record
                paired_solutions[matrix_free] = solutions
                print(
                    f"grid={size} rhs={columns} repeat={repeat} "
                    f"mode={record['mode']} cpu={record['cpu_seconds']:.6f}s",
                    flush=True,
                )

            sparse_record = paired_records[False]
            matrix_free_record = paired_records[True]
            velocity_errors = []
            pressure_errors = []
            for actual, expected in zip(
                paired_solutions[True], paired_solutions[False], strict=True
            ):
                velocity_errors.append(relative_error(actual[0], expected[0]))
                pressure_errors.append(relative_error(actual[1], expected[1]))
            comparisons.append(
                {
                    "grid": size,
                    "right_hand_sides": columns,
                    "repeat": repeat,
                    "sparse_over_matrix_free_cpu": (
                        sparse_record["cpu_seconds"]
                        / matrix_free_record["cpu_seconds"]
                    ),
                    "sparse_over_matrix_free_wall": (
                        sparse_record["wall_seconds"]
                        / matrix_free_record["wall_seconds"]
                    ),
                    "iteration_traces_equal": (
                        sparse_record["iteration_trace"]
                        == matrix_free_record["iteration_trace"]
                    ),
                    "max_velocity_relative_error": max(velocity_errors),
                    "max_pressure_relative_error": max(pressure_errors),
                }
            )
    return records, comparisons


def summarize(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for size in config["grids"]:
        for columns in config["right_hand_sides"]:
            group = [
                item
                for item in comparisons
                if item["grid"] == size and item["right_hand_sides"] == columns
            ]
            item: dict[str, Any] = {
                "grid": size,
                "right_hand_sides": columns,
                "cpu_speedup_median": float(
                    np.median(
                        [entry["sparse_over_matrix_free_cpu"] for entry in group]
                    )
                ),
                "wall_speedup_median": float(
                    np.median(
                        [entry["sparse_over_matrix_free_wall"] for entry in group]
                    )
                ),
                "iteration_traces_equal": all(
                    entry["iteration_traces_equal"] for entry in group
                ),
                "max_velocity_relative_error": max(
                    entry["max_velocity_relative_error"] for entry in group
                ),
                "max_pressure_relative_error": max(
                    entry["max_pressure_relative_error"] for entry in group
                ),
            }
            for mode in ("sparse", "matrix_free_reuse"):
                mode_records = [
                    record
                    for record in records
                    if record["grid"] == size
                    and record["right_hand_sides"] == columns
                    and record["mode"] == mode
                ]
                item[f"{mode}_cpu_seconds_median"] = float(
                    np.median([record["cpu_seconds"] for record in mode_records])
                )
                item[f"{mode}_wall_seconds_median"] = float(
                    np.median([record["wall_seconds"] for record in mode_records])
                )
            summary.append(item)
    return summary


def run_benchmark(config: dict[str, Any], output: Path) -> Path:
    jit_warmup_wall_start = perf_counter()
    jit_warmup_cpu_start = process_time()
    warmup_seconds = warmup()
    jit_warmup_cpu = process_time() - jit_warmup_cpu_start
    jit_warmup_wall = perf_counter() - jit_warmup_wall_start
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    with set_workers(config["fft_workers"]):
        for size in config["grids"]:
            grid_records, grid_comparisons = run_grid(config, size)
            records.extend(grid_records)
            comparisons.extend(grid_comparisons)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance(),
        "config": config,
        "jit_warmup_reported_seconds": warmup_seconds,
        "jit_warmup_cpu_seconds": jit_warmup_cpu,
        "jit_warmup_wall_seconds": jit_warmup_wall,
        "records": records,
        "comparisons": comparisons,
        "summary": summarize(config, records, comparisons),
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
        identity + "-matrix-free-stokes"
    ) / "timings.json"
    completed = run_benchmark(config, output)
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
