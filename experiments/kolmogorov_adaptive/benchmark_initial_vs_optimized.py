"""Compare the pre-decoupling solver with the fully optimized solver.

Purpose: measure end-to-end CPU time for ten fixed or random Kolmogorov-flow
steps using the original sparse DirectStokes implementation and the current
tau-decoupled, matrix-free TensorStokes implementation.  The grid, initial
condition, forcing, time schemes, step schedules, tolerances, Python
environment, and thread count are fixed.  Backend setup is included; imports
and Numba warmup are excluded.  Improvement requires lower paired CPU time
without a material final-state or divergence change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path("/private/tmp/mac-mrsav-baseline-13d8a26")
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
    "primary_timing": "process_time_seconds_including_backend_setup_excluding_jit",
}
SCHEMES = ("sdirk2", "sdirk2_mrsav", "sdirk3", "sdirk3_mrsav")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def schedules(config: dict[str, Any]) -> dict[str, np.ndarray]:
    fixed = np.full(config["steps"], config["fixed_step"])
    rng = np.random.default_rng(config["random_seed"])
    low, high = config["random_multiplier_interval"]
    random = rng.uniform(low, high, size=config["steps"])
    random *= fixed.sum() / random.sum()
    return {"fixed": fixed, "random": random}


def import_scheme(name: str) -> Any:
    module = importlib.import_module(f"solver.schemes.{name}")
    class_name = {
        "sdirk2": "SDIRK2",
        "sdirk2_mrsav": "SDIRK2MRSAV",
        "sdirk3": "SDIRK3",
        "sdirk3_mrsav": "SDIRK3MRSAV",
    }[name]
    return getattr(module, class_name)


def worker(
    project_root: Path,
    implementation: str,
    scheme_name: str,
    schedule_name: str,
    record_path: Path,
    velocity_path: Path,
) -> None:
    sys.path.insert(0, str(project_root))
    model_module = importlib.import_module("experiments.kolmogorov_adaptive.model")
    integrate = importlib.import_module("solver.integrate").integrate
    MACGrid = importlib.import_module("solver.mac.grid").MACGrid
    kernels = importlib.import_module("solver.mac.kernels")
    DirectStokes = importlib.import_module("solver.mac.stokes").DirectStokes
    MACNavierStokes = importlib.import_module("solver.mac_ns").MACNavierStokes

    kernels.warmup()
    grid = MACGrid(CONFIG["nx"], CONFIG["ny"], CONFIG["lx"], CONFIG["ly"])
    initial = model_module.initial_velocity(
        grid, CONFIG["epsilon"], CONFIG["initial_modes"]
    )
    load = model_module.force(grid, CONFIG["forcing_m"])
    model = MACNavierStokes(grid, CONFIG["nu"], cache_size=CONFIG["cache_size"])
    model.force = lambda _: load
    model.nonlinear(initial)

    setup_cpu_start = process_time()
    setup_wall_start = perf_counter()
    if implementation == "initial":
        backend = DirectStokes(
            model.ops,
            cache_size=CONFIG["cache_size"],
            tolerance=CONFIG["stokes_tolerance"],
        )
    else:
        TensorStokes = importlib.import_module(
            "solver.mac.tensor_stokes"
        ).TensorStokes
        backend = TensorStokes(
            model.ops,
            tolerance=CONFIG["stokes_tolerance"],
            krylov_tolerance=CONFIG["krylov_tolerance"],
            max_iterations=CONFIG["max_iterations"],
            matrix_free_operators=True,
            reuse_transform_buffers=True,
        )
    setup_cpu = process_time() - setup_cpu_start
    setup_wall = perf_counter() - setup_wall_start
    model.backend = backend

    constructor = import_scheme(scheme_name)
    scheme = constructor(CONFIG["gamma"]) if scheme_name.endswith("mrsav") else constructor()
    state = model.state(0.0, initial.copy())
    step_schedule = schedules(CONFIG)[schedule_name]
    integration_cpu_start = process_time()
    integration_wall_start = perf_counter()
    result = integrate(model, scheme, state, step_schedule)
    integration_cpu = process_time() - integration_cpu_start
    integration_wall = perf_counter() - integration_wall_start
    if result.status != "complete":
        raise RuntimeError(
            f"{implementation}/{scheme_name}/{schedule_name}: {result.error}"
        )

    velocity = model.vector(result.final)
    np.save(velocity_path, velocity)
    record: dict[str, Any] = {
        "implementation": implementation,
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
    }
    if implementation == "initial":
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
            schur_iterations_per_rhs=(
                backend.total_iterations / backend.right_hand_sides
            ),
        )
    write_json(record_path, record)


def source_record(project_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain", "--", "solver"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    hasher = hashlib.sha256()
    for path in sorted((project_root / "solver").rglob("*.py")):
        hasher.update(str(path.relative_to(project_root)).encode())
        hasher.update(path.read_bytes())
    return {
        "project_root": str(project_root),
        "git_commit": commit,
        "solver_dirty": dirty,
        "solver_code_sha256": hasher.hexdigest(),
    }


def median(records: list[dict[str, Any]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def summarize(
    records: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    summary: list[dict[str, Any]] = []
    for scheme_name in SCHEMES:
        for schedule_name in ("fixed", "random"):
            groups = {
                implementation: [
                    record
                    for record in records
                    if record["implementation"] == implementation
                    and record["scheme"] == scheme_name
                    and record["schedule"] == schedule_name
                ]
                for implementation in ("initial", "optimized")
            }
            paired = [
                comparison
                for comparison in comparisons
                if comparison["scheme"] == scheme_name
                and comparison["schedule"] == schedule_name
            ]
            initial_cpu = median(groups["initial"], "total_cpu_seconds")
            optimized_cpu = median(groups["optimized"], "total_cpu_seconds")
            summary.append(
                {
                    "scheme": scheme_name,
                    "schedule": schedule_name,
                    "initial_total_cpu_seconds_median": initial_cpu,
                    "optimized_total_cpu_seconds_median": optimized_cpu,
                    "median_ratio_of_cpu_times": float(
                        np.median([item["cpu_speedup"] for item in paired])
                    ),
                    "cpu_reduction_percent_from_medians": (
                        100.0 * (1.0 - optimized_cpu / initial_cpu)
                    ),
                    "max_final_velocity_relative_error": max(
                        item["final_velocity_relative_error"] for item in paired
                    ),
                    "max_initial_divergence": max(
                        record["max_divergence"] for record in groups["initial"]
                    ),
                    "max_optimized_divergence": max(
                        record["max_divergence"] for record in groups["optimized"]
                    ),
                    "initial_factorizations": sorted(
                        {record["factorizations"] for record in groups["initial"]}
                    ),
                    "optimized_schur_iterations_per_rhs_median": median(
                        groups["optimized"], "schur_iterations_per_rhs"
                    ),
                }
            )
    initial_sum = sum(item["initial_total_cpu_seconds_median"] for item in summary)
    optimized_sum = sum(item["optimized_total_cpu_seconds_median"] for item in summary)
    aggregate = {
        "sum_of_case_median_initial_cpu_seconds": initial_sum,
        "sum_of_case_median_optimized_cpu_seconds": optimized_sum,
        "aggregate_cpu_speedup": initial_sum / optimized_sum,
        "aggregate_cpu_reduction_percent": 100.0 * (1.0 - optimized_sum / initial_sum),
        "geometric_mean_case_speedup": math.exp(
            sum(math.log(item["median_ratio_of_cpu_times"]) for item in summary)
            / len(summary)
        ),
    }
    return summary, aggregate


def run_benchmark(
    baseline_root: Path, current_root: Path, repeats: int, output: Path
) -> Path:
    if not (baseline_root / "solver/mac/stokes.py").exists():
        raise FileNotFoundError(f"Baseline worktree is missing: {baseline_root}")
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_VARIABLES})
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    roots = {"initial": baseline_root, "optimized": current_root}

    with tempfile.TemporaryDirectory(prefix="mac-initial-vs-optimized-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(1, repeats + 1):
            order = ("initial", "optimized") if repeat % 2 else ("optimized", "initial")
            for scheme_name in SCHEMES:
                for schedule_name in ("fixed", "random"):
                    pair: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
                    for implementation in order:
                        stem = f"{repeat}-{scheme_name}-{schedule_name}-{implementation}"
                        record_path = temporary_root / f"{stem}.json"
                        velocity_path = temporary_root / f"{stem}.npy"
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--project-root",
                            str(roots[implementation]),
                            "--implementation",
                            implementation,
                            "--scheme",
                            scheme_name,
                            "--schedule",
                            schedule_name,
                            "--record",
                            str(record_path),
                            "--velocity",
                            str(velocity_path),
                        ]
                        subprocess.run(
                            command,
                            cwd=roots[implementation],
                            env=environment,
                            check=True,
                        )
                        record = json.loads(record_path.read_text())
                        record["repeat"] = repeat
                        records.append(record)
                        pair[implementation] = (record, np.load(velocity_path))

                    initial_record, initial_velocity = pair["initial"]
                    optimized_record, optimized_velocity = pair["optimized"]
                    denominator = np.linalg.norm(initial_velocity)
                    comparison = {
                        "repeat": repeat,
                        "scheme": scheme_name,
                        "schedule": schedule_name,
                        "cpu_speedup": (
                            initial_record["total_cpu_seconds"]
                            / optimized_record["total_cpu_seconds"]
                        ),
                        "wall_speedup": (
                            initial_record["total_wall_seconds"]
                            / optimized_record["total_wall_seconds"]
                        ),
                        "final_velocity_relative_error": float(
                            np.linalg.norm(optimized_velocity - initial_velocity)
                            / denominator
                        ),
                    }
                    comparisons.append(comparison)
                    print(
                        f"repeat={repeat} scheme={scheme_name} schedule={schedule_name} "
                        f"initial_cpu={initial_record['total_cpu_seconds']:.6f}s "
                        f"optimized_cpu={optimized_record['total_cpu_seconds']:.6f}s "
                        f"speedup={comparison['cpu_speedup']:.3f}x",
                        flush=True,
                    )

    summary, aggregate = summarize(records, comparisons)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": __doc__.split("\n\n", 1)[1].strip(),
        "versions": {
            "initial": source_record(baseline_root),
            "optimized": source_record(current_root),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "numba")
            },
            "threads": {name: environment[name] for name in THREAD_VARIABLES},
        },
        "config": {**CONFIG, "repeats": repeats},
        "fixed_steps": schedules(CONFIG)["fixed"].tolist(),
        "random_steps": schedules(CONFIG)["random"].tolist(),
        "records": records,
        "comparisons": comparisons,
        "summary": summary,
        "aggregate": aggregate,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--current-root", type=Path, default=PROJECT)
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--implementation", choices=("initial", "optimized"))
    parser.add_argument("--scheme", choices=SCHEMES)
    parser.add_argument("--schedule", choices=("fixed", "random"))
    parser.add_argument("--record", type=Path)
    parser.add_argument("--velocity", type=Path)
    arguments = parser.parse_args()

    if arguments.worker:
        required = (
            arguments.project_root,
            arguments.implementation,
            arguments.scheme,
            arguments.schedule,
            arguments.record,
            arguments.velocity,
        )
        if any(value is None for value in required):
            parser.error("worker mode requires all worker arguments")
        worker(
            arguments.project_root,
            arguments.implementation,
            arguments.scheme,
            arguments.schedule,
            arguments.record,
            arguments.velocity,
        )
        return

    if arguments.repeats < 1:
        raise ValueError("repeats must be positive")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    identity += "-" + uuid4().hex[:8]
    output = arguments.output or PROJECT / "reports/timings" / (
        identity + "-initial-vs-optimized-10-step"
    ) / "timings.json"
    completed = run_benchmark(
        arguments.baseline_root.resolve(),
        arguments.current_root.resolve(),
        arguments.repeats,
        output,
    )
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
