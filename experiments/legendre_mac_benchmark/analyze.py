"""Read one checksum-verified benchmark run; report time versus physical error."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.legendre_mac_benchmark.run import relative_error
from experiments.records import PROJECT, digest, write_json

LABELS = {"mac_direct": "MAC / sparse LU", "mac_tensor": "MAC / tensor Schur-CG",
          "legendre_lu": "Legendre / sparse LU"}
COLORS = {"mac_direct": "#0072B2", "mac_tensor": "#009E73", "legendre_lu": "#D55E00"}
MARKERS = {"mac_direct": "o", "mac_tensor": "s", "legendre_lu": "^"}


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    """Aggregate repeats and select the fastest measured member meeting each error gate."""
    directory = directory.resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analysis requires a complete run")
    for name, expected in manifest["artifacts"].items():
        if digest(directory/name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    config = json.loads((directory/"config.json").read_text())
    records = json.loads((directory/"results.json").read_text())["records"]
    if len(records) != manifest["planned_cases"]:
        raise ValueError("Incomplete case matrix")
    with np.load(directory/"final_fields.npz", allow_pickle=False) as fields:
        for record in records:
            if record["repeat"] == 0:
                error = relative_error(config, record["backend"], record["size"],
                                       fields[record["field_key"]], record["final_time"],
                                       config["error_quadrature_refined"])
                if not np.isclose(error, record["relative_L2_refined"], rtol=1e-10, atol=1e-15):
                    raise ValueError("Saved field error disagrees with timing record")
        backend_differences = []
        for record in records:
            if record["backend"] == "mac_tensor":
                direct = record["field_key"].replace("mac_tensor_", "mac_direct_", 1)
                error = np.linalg.norm(fields[record["field_key"]]-fields[direct])/np.linalg.norm(fields[direct])
                backend_differences.append(float(error))
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for record in records:
        group_key = (record["backend"], record["size"], record["scheme"], record["schedule"])
        grouped.setdefault(group_key, []).append(record)
    summaries = []
    for (backend, size, scheme, schedule), members in grouped.items():
        if len(members) != config["repeats"] or {m["repeat"] for m in members} != set(range(config["repeats"])):
            raise ValueError("Missing or duplicate repeat")
        row = {"backend": backend, "size": size, "scheme": scheme, "schedule": schedule,
               "velocity_dofs": members[0]["velocity_dofs"], "pressure_dofs": size**2-1,
               "repeats": len(members), "relative_L2": max(m["relative_L2_refined"] for m in members),
               "quadrature_passed": all(m["quadrature_passed"] for m in members),
               "quadrature_relative_difference": max(m["quadrature_relative_difference"] for m in members),
               "factorizations": members[0]["factorizations"],
               "max_stage_residual": max(m["max_stage_residual"] for m in members),
               "max_stage_continuity_residual": max(m["max_stage_continuity_residual"] for m in members),
               "max_physical_divergence": max(m["max_physical_divergence"] for m in members)}
        for key in ("setup_cpu_seconds", "integration_cpu_seconds", "total_cpu_seconds", "total_wall_seconds",
                    "nonlinear_cpu_seconds", "diagnostics_cpu_seconds", "cached_stokes_cpu_seconds"):
            row[key] = float(np.median([member[key] for member in members]))
        row["total_cpu_min"] = min(m["total_cpu_seconds"] for m in members)
        row["total_cpu_max"] = max(m["total_cpu_seconds"] for m in members)
        summaries.append(row)
    accuracy = []
    for scheme in config["schemes"]:
        for schedule in ("fixed", "variable"):
            for threshold in config["accuracy_thresholds"]:
                for backend in LABELS:
                    eligible = [r for r in summaries if r["scheme"] == scheme and r["schedule"] == schedule
                                and r["backend"] == backend and r["quadrature_passed"]
                                and r["relative_L2"] <= threshold]
                    best = min(eligible, key=lambda r: r["total_cpu_seconds"]) if eligible else None
                    accuracy.append({"scheme": scheme, "schedule": schedule, "threshold": threshold,
                                     "backend": backend, "best_size": best["size"] if best else None,
                                     "relative_L2": best["relative_L2"] if best else None,
                                     "total_cpu_seconds": best["total_cpu_seconds"] if best else None})
    report = root / "reports/legendre_mac_benchmark" / manifest["id"]
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/"analysis.json", {"run": str(directory), "manifest_sha256": digest(directory/"manifest.json"),
                                       "summaries": summaries, "accuracy_gates": accuracy,
                                       "mac_tensor_vs_direct_relative_difference_max": max(backend_differences),
                                       "all_quadrature_checks_passed": all(r["quadrature_passed"] for r in summaries)})
    for name, rows in (("timings.csv", summaries), ("accuracy_gates.csv", accuracy)):
        with (report/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    figure, axes = plt.subplots(len(config["schemes"]), 2, figsize=(10, 3.6*len(config["schemes"])),
                                squeeze=False, layout="constrained")
    for i, scheme in enumerate(config["schemes"]):
        for j, schedule in enumerate(("fixed", "variable")):
            axis = axes[i, j]
            for backend in LABELS:
                rows = sorted([r for r in summaries if r["scheme"] == scheme and r["schedule"] == schedule
                               and r["backend"] == backend], key=lambda r: r["size"])
                axis.loglog([r["relative_L2"] for r in rows], [r["total_cpu_seconds"] for r in rows],
                            color=COLORS[backend], marker=MARKERS[backend], label=LABELS[backend])
                for r in rows:
                    offset = (-22, -10) if backend == "mac_direct" else (4, 4)
                    axis.annotate(str(r["size"]), (r["relative_L2"], r["total_cpu_seconds"]),
                                  xytext=offset, textcoords="offset points", fontsize=8)
            scheme_label = "SDIRK3-mr-ccSAV" if scheme.endswith("mrsav") else "SDIRK3"
            axis.set(xlabel="Relative velocity L2 error (common physical quadrature)",
                     ylabel=f"Setup + {config['steps']} steps CPU time (s)",
                     title=f"{scheme_label} / {schedule} steps")
            axis.margins(x=.08, y=.15)
            axis.grid(True, which="both", alpha=.2)
            axis.legend(fontsize=8)
    figure.savefig(report/"time_accuracy.png", dpi=180)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(f"REPORT={analyze(args.run)}")


if __name__ == "__main__":
    main()
