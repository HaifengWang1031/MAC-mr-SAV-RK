"""Compare DirectStokes and TensorStokes in the actual adaptive driver.

Purpose: verify that backend replacement preserves short-time I/PI acceptance
decisions and quantify CPU time.  The grid, initial data, tolerances, controller,
scheme and duration are fixed; only the Stokes backend varies.  A successful
comparison has identical accept/reject decisions, nearly identical proposed
steps and final velocity, and lower TensorStokes CPU time.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.kolmogorov_adaptive.run import (
    SCHEMES,
    config_check,
    make_scheme,
    model_and_initial,
)
from experiments.workflow import PROJECT, provenance, write_json
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.mac.kernels import warmup
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes


DEFAULT_DURATION = 0.02
DEFAULT_REPEATS = 3
CONTROLLERS = {"I": IController, "PI": PIController}


def benchmark_config(duration: float) -> dict[str, Any]:
    path = PROJECT / "experiments/kolmogorov_adaptive/configs/production.json"
    config = json.loads(path.read_text())
    config.update(T=duration, output_every=duration)
    config_check(config)
    return config


def controller(config: dict[str, Any], name: str, scheme_name: str) -> IController:
    return CONTROLLERS[name](
        atol=config["atol_velocity"],
        rtol=config["rtol_velocity"],
        safety=config["safety"],
        min_step=config["min_step"],
        max_step=config["max_step"],
        estimator_order=2 if scheme_name.startswith("sdirk2") else 3,
    )


def run_case(
    config: dict[str, Any], scheme_name: str, controller_name: str, backend_name: str, repeat: int
) -> tuple[dict[str, Any], np.ndarray]:
    case = {**config, "stokes_backend": backend_name}
    setup_start = process_time()
    model, initial = model_and_initial(case)
    setup_cpu = process_time() - setup_start
    integration_start = process_time()
    result = integrate_adaptive(
        model,
        make_scheme(scheme_name, config["gamma"]),
        initial,
        config["T"],
        config["initial_step"],
        controller=controller(config, controller_name, scheme_name),
    )
    integration_cpu = process_time() - integration_start
    if result.status != "complete":
        raise RuntimeError(
            f"{backend_name}/{scheme_name}/{controller_name} failed: {result.error}"
        )
    record: dict[str, Any] = {
        "repeat": repeat,
        "backend": backend_name,
        "scheme": scheme_name,
        "controller": controller_name,
        "setup_cpu_seconds": setup_cpu,
        "integration_cpu_seconds": integration_cpu,
        "total_cpu_seconds": setup_cpu + integration_cpu,
        "accepted_steps": len(result.times) - 1,
        "rejected_trials": len(result.attempts) - len(result.times) + 1,
        "attempt_steps": [float(item["step"]) for item in result.attempts],
        "attempt_accepted": [bool(item["accepted"]) for item in result.attempts],
        "attempt_errors": [float(item["error"]) for item in result.attempts],
        "max_divergence": max(item["divergence_inf"] for item in result.diagnostics),
        "final_kinetic": result.diagnostics[-1]["kinetic"],
    }
    if isinstance(model.backend, DirectStokes):
        record.update(
            factorizations=model.backend.factorizations,
            factorization_seconds=model.backend.factorization_seconds,
            linear_solve_seconds=model.backend.solve_seconds,
        )
    elif isinstance(model.backend, TensorStokes):
        record.update(
            stokes_solves=model.backend.solves,
            schur_right_hand_sides=model.backend.right_hand_sides,
            schur_iterations=model.backend.total_iterations,
            schur_iterations_per_rhs=(
                model.backend.total_iterations / model.backend.right_hand_sides
            ),
        )
    return record, model.vector(result.final)


def run_benchmark(config: dict[str, Any], repeats: int, output: Path) -> Path:
    records: list[dict[str, Any]] = []
    jit_warmup_seconds = warmup()
    for repeat in range(1, repeats + 1):
        for scheme_name in SCHEMES:
            for controller_name in CONTROLLERS:
                direct_record, direct_velocity = run_case(
                    config, scheme_name, controller_name, "direct", repeat
                )
                tensor_record, tensor_velocity = run_case(
                    config, scheme_name, controller_name, "tensor", repeat
                )
                if tensor_record["attempt_accepted"] != direct_record["attempt_accepted"]:
                    raise RuntimeError(
                        f"Acceptance mismatch for {scheme_name}/{controller_name}"
                    )
                direct_steps = np.asarray(direct_record["attempt_steps"])
                tensor_steps = np.asarray(tensor_record["attempt_steps"])
                step_difference = float(
                    np.max(np.abs(tensor_steps - direct_steps)) / np.max(direct_steps)
                )
                velocity_difference = float(
                    np.linalg.norm(tensor_velocity - direct_velocity)
                    / np.linalg.norm(direct_velocity)
                )
                tensor_record.update(
                    attempt_step_relative_difference=step_difference,
                    final_velocity_relative_difference=velocity_difference,
                )
                direct_record.update(
                    attempt_step_relative_difference=0.0,
                    final_velocity_relative_difference=0.0,
                )
                if step_difference > 1e-6 or velocity_difference > 1e-9:
                    raise RuntimeError(
                        f"Backend mismatch for {scheme_name}/{controller_name}: "
                        f"step={step_difference:.3e}, velocity={velocity_difference:.3e}"
                    )
                records.extend((direct_record, tensor_record))
                print(
                    f"repeat={repeat} scheme={scheme_name} controller={controller_name} "
                    f"direct={direct_record['total_cpu_seconds']:.6f}s "
                    f"tensor={tensor_record['total_cpu_seconds']:.6f}s",
                    flush=True,
                )

    summary = []
    for scheme_name in SCHEMES:
        for controller_name in CONTROLLERS:
            groups = {
                backend: [
                    record
                    for record in records
                    if record["scheme"] == scheme_name
                    and record["controller"] == controller_name
                    and record["backend"] == backend
                ]
                for backend in ("direct", "tensor")
            }
            direct_cpu = float(np.median([r["total_cpu_seconds"] for r in groups["direct"]]))
            tensor_cpu = float(np.median([r["total_cpu_seconds"] for r in groups["tensor"]]))
            summary.append(
                {
                    "scheme": scheme_name,
                    "controller": controller_name,
                    "direct_total_cpu_seconds_median": direct_cpu,
                    "tensor_total_cpu_seconds_median": tensor_cpu,
                    "direct_over_tensor_speedup": direct_cpu / tensor_cpu,
                    "accepted_steps": sorted({r["accepted_steps"] for r in groups["tensor"]}),
                    "rejected_trials": sorted({r["rejected_trials"] for r in groups["tensor"]}),
                    "tensor_schur_iterations_per_rhs_median": float(
                        np.median([r["schur_iterations_per_rhs"] for r in groups["tensor"]])
                    ),
                }
            )
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Adaptive I/PI backend equivalence and CPU comparison",
        "source": provenance(),
        "config": config,
        "repeats": repeats,
        "jit_warmup_seconds": jit_warmup_seconds,
        "records": records,
        "summary": summary,
        "max_attempt_step_relative_difference": max(
            record["attempt_step_relative_difference"] for record in records
        ),
        "max_final_velocity_relative_difference": max(
            record["final_velocity_relative_difference"] for record in records
        ),
        "max_tensor_divergence": max(
            record["max_divergence"] for record in records if record["backend"] == "tensor"
        ),
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.duration <= 0 or arguments.repeats < 1:
        raise ValueError("duration and repeats must be positive")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = arguments.output or PROJECT / "reports/timings" / (
        identity + "-adaptive-stokes-backends"
    ) / "timings.json"
    completed = run_benchmark(benchmark_config(arguments.duration), arguments.repeats, output)
    print(f"report={completed.resolve()}")


if __name__ == "__main__":
    main()
