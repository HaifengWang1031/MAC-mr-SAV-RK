"""Measure FFT temporary-buffer reuse in matrix-free TensorStokes.

Purpose: isolate the effect of allowing SciPy's second forward transform and
both inverse transforms to overwrite disposable temporaries.  Matrix-free MAC
operators, scalar CG, grids, right-hand sides, viscosity sequence and tolerances
remain fixed.  Timings exclude backend setup and Numba warmup.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.fft import set_workers

from experiments.kolmogorov_adaptive.benchmark_matrix_free_stokes import (
    CONFIG as BASE_CONFIG,
    relative_error,
    run_solver,
)
from experiments.workflow import PROJECT, provenance, write_json
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.operators import MACOperators
from solver.mac.tensor_stokes import TensorStokes


CONFIG = {
    **BASE_CONFIG,
    "repeats": 5,
    "primary_timing": "process_time_seconds_excluding_backend_setup_and_jit",
}


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
        for reuse_buffers in (False, True):
            warm = TensorStokes(
                operators,
                matrix_free_operators=True,
                reuse_transform_buffers=reuse_buffers,
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
            for reuse_buffers in modes:
                record, solutions = run_solver(
                    config,
                    operators,
                    right_hand_sides,
                    viscosities,
                    True,
                    repeat,
                    reuse_transform_buffers=reuse_buffers,
                )
                record.update(grid=size, right_hand_sides=columns, solves=count)
                records.append(record)
                paired_records[reuse_buffers] = record
                paired_solutions[reuse_buffers] = solutions
                print(
                    f"grid={size} rhs={columns} repeat={repeat} "
                    f"mode={record['mode']} cpu={record['cpu_seconds']:.6f}s",
                    flush=True,
                )

            baseline_record = paired_records[False]
            reused_record = paired_records[True]
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
                    "baseline_over_reuse_cpu": (
                        baseline_record["cpu_seconds"] / reused_record["cpu_seconds"]
                    ),
                    "baseline_over_reuse_wall": (
                        baseline_record["wall_seconds"]
                        / reused_record["wall_seconds"]
                    ),
                    "iteration_traces_equal": (
                        baseline_record["iteration_trace"]
                        == reused_record["iteration_trace"]
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
                    np.median([entry["baseline_over_reuse_cpu"] for entry in group])
                ),
                "wall_speedup_median": float(
                    np.median([entry["baseline_over_reuse_wall"] for entry in group])
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
            for mode in ("matrix_free", "matrix_free_reuse"):
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
    jit_wall_start = perf_counter()
    jit_cpu_start = process_time()
    warmup_seconds = warmup()
    jit_cpu = process_time() - jit_cpu_start
    jit_wall = perf_counter() - jit_wall_start
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
        "jit_warmup_cpu_seconds": jit_cpu,
        "jit_warmup_wall_seconds": jit_wall,
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
        identity + "-transform-buffer-reuse"
    ) / "timings.json"
    completed = run_benchmark(config, output)
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
