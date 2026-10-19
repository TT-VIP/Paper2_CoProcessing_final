"""Analyze calibration runs and master-problem progress for Yue decomposition.

Place this script alongside the ``Solutions`` folder, then run:

    python Analyze_CalibrationStudy_extended.py

Outputs are saved to ``Solutions/calib/analysis``. Optional flags:

    --no-plots         Export CSV tables only.
    --all-run-plots    Also create detailed incumbent/bound plots for every run.
    --study-dir PATH   Override the default calibration folder.

The script reads exported JSON files only; it does not solve optimization models.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STUDY_DIR = ROOT / "Solutions" / "SOS1"
EXPECTED_RUNS = 45
EXPECTED_INSTANCES = 45
EXPECTED_CONFIGURATIONS = 1
EXPECTED_THREADS = 16

RUN_NUMERIC_COLS = (
    "big_m", "cutoff_tolerance", "threads", "runtime_s", "time_limit_s",
    "iterations", "completed_iterations", "lb", "ub", "mp_runtime_s",
    "sp1_runtime_s", "sp2_runtime_s", "recorded_iteration_wall_s",
)
ITER_NUMERIC_COLS = (
    "iteration", "mp_obj", "mp_bound", "mp_mip_gap", "lb", "ub",
    "mp_runtime", "sp1_runtime", "sp2_runtime", "total_iteration_time", "total_time",
)



# Status labels as exported by Decomposition_Algorithm.
# Both known spellings of the no-bilevel-solution status are supported.
STATUS_LABELS = {
    "OPTIMAL_PROVEN": "proven",
    "SUBOPTIMAL_MPSP2_INCUMBENTS_MATCH": "incumbent_match",
    "FEASIBLE_SUBOPTIMAL": "feasible_suboptimal",
    "MP_INFEASIBLE_OR_NO_SOLUTION": "mp_no_solution",
    "NO_BILEVEL_FEASIBLE_SOLUTION_FOUND": "no_bilevel_feasible",
    "NO_BILEVEL_FEASIBLE_SOLUTION": "no_bilevel_feasible",
    "NUMERICAL_BOUND_INCONSISTENCY": "numerical_inconsistency",
}
STATUS_COUNT_COLUMNS = (
    "n_proven", "n_incumbent_match", "n_feasible_suboptimal",
    "n_mp_no_solution", "n_no_bilevel_feasible", "n_numerical_inconsistency",
)


def add_status_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original status and create transparent status indicators."""
    runs = runs.copy()
    unknown = set(runs["status"].dropna().unique()) - set(STATUS_LABELS)
    if unknown:
        raise ValueError(f"Unknown decomposition status values: {sorted(unknown)}")

    runs["status_group"] = runs["status"].map(STATUS_LABELS)
    for name in STATUS_COUNT_COLUMNS:
        runs[name] = runs["status_group"].eq(name.removeprefix("n_"))

    # Use exported feasible-solution information, not merely a status label.
    runs["any_feasible"] = runs["feasible"] & ~runs["numerical_issue"]
    runs["no_feasible"] = ~runs["feasible"]
    runs["unproven_feasible"] = runs["any_feasible"] & ~runs["proven"]
    runs["unproven_feasible_gap_pct"] = runs["relative_gap_pct"].where(
        runs["unproven_feasible"]
    )
    runs["feasible_runtime_s"] = runs["runtime_s"].where(runs["any_feasible"])

    # A status/solution disagreement may indicate inconsistent exported metadata.
    expects_feasible = runs["status_group"].isin(
        ["proven", "incumbent_match", "feasible_suboptimal"]
    )
    expects_no_feasible = runs["status_group"].isin(
        ["mp_no_solution", "no_bilevel_feasible"]
    )
    inconsistent = (expects_feasible & ~runs["feasible"]) | (
        expects_no_feasible & runs["feasible"]
    )
    if inconsistent.any():
        import warnings
        warnings.warn(
            f"{int(inconsistent.sum())} run(s) have status/feasibility mismatches; "
            "inspect status and has_feasible_solution fields.",
            stacklevel=2,
        )
    return runs


def status_aggregations() -> dict:
    """Reusable status counts and conditional quality statistics."""
    return {
        **{name: (name, "sum") for name in STATUS_COUNT_COLUMNS},
        "any_feasible": ("any_feasible", "sum"),
        "no_feasible": ("no_feasible", "sum"),
        "feasible_unproven": ("unproven_feasible", "sum"),
        "median_unproven_feasible_gap_pct": (
            "unproven_feasible_gap_pct", "median"
        ),
        "median_feasible_runtime_s": ("feasible_runtime_s", "median"),
    }


def summarize_statuses(runs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Status composition with denominators, suitable for paired comparison."""
    out = runs.groupby(group_cols, as_index=False, dropna=False).agg(
        n_runs=("status", "size"),
        **status_aggregations(),
    )
    out["feasible_pct"] = 100 * out["any_feasible"] / out["n_runs"]
    out["proven_pct"] = 100 * out["n_proven"] / out["n_runs"]
    return out


def save_status_tables(runs: pd.DataFrame, out_dir: Path) -> None:
    """Write configuration-, size-, regime-, and instance-level status reports."""
    groupings = {
        "calibration_status_by_configuration.csv": ["formulation", "configuration"],
        "calibration_status_by_size.csv": ["formulation", "configuration", "size"],
        "calibration_status_by_regime.csv": ["formulation", "configuration", "regime"],
        "calibration_status_formulation_by_size.csv": ["formulation", "size"],
    }
    for filename, columns in groupings.items():
        summarize_statuses(runs, columns).to_csv(out_dir / filename, index=False)
    runs.pivot(
        index=["size", "regime", "instance"],
        columns="configuration",
        values="status",
    ).to_csv(out_dir / "calibration_status_by_instance.csv")


def load_results(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read runs and iteration histories into separate relational tables."""
    paths = sorted(study_dir.rglob("SOL_*.json"))
    if not paths:
        raise FileNotFoundError(f"No SOL_*.json files found in {study_dir}")

    runs: list[dict] = []
    iterations: list[dict] = []

    for path in paths:
        with path.open(encoding="utf-8") as stream:
            obj = json.load(stream)

        metadata = obj["metadata"]
        config = obj["algorithm_config"]
        result = obj["result"]
        history = obj.get("iteration_history", [])

        key = {
            "instance": metadata["instance_name"],
            "size": metadata["size_class"],
            "regime": metadata["structural_regime"],
            "configuration": metadata["method_tag"],
            "formulation": config["complementarity_formulation"],
            "strengthening": config.get("primal_dual_strengthening"),
            "big_m": config.get("bigM_duals_unrestricted"),
            "cutoff_tolerance": config.get("cutoff_bound_tolerance"),
            "threads": config.get("threads"),
        }

        def cumulative_runtime(field: str) -> float:
            return sum(float(rec[field]) for rec in history if rec.get(field) is not None)

        runs.append({
            **key,
            "status": result["status"],
            "termination_reason": result.get("termination_reason"),
            "runtime_s": result.get("total_solution_time"),
            "time_limit_s": config.get("total_time_limit"),
            "iterations": result.get("iterations"),
            "completed_iterations": result.get("completed_iterations", len(history)),
            "history_records": len(history),
            "lb": result.get("lower_bound"),
            "ub": result.get("upper_bound"),
            "has_feasible_solution": obj.get("best_solution") is not None,
            "mp_runtime_s": cumulative_runtime("mp_runtime"),
            "sp1_runtime_s": cumulative_runtime("sp1_runtime"),
            "sp2_runtime_s": cumulative_runtime("sp2_runtime"),
            "recorded_iteration_wall_s": cumulative_runtime("total_iteration_time"),
            "source_file": str(path),
        })
        for rec in history:
            iterations.append({
                **key,
                "iteration": rec["iteration"],
                "mp_obj": rec.get("mp_obj"),
                "mp_bound": rec.get("mp_bound"),
                "mp_mip_gap": rec.get("mp_mip_gap"),
                "lb": rec.get("lb"),
                "ub": rec.get("ub"),
                "mp_runtime": rec.get("mp_runtime"),
                "sp1_runtime": rec.get("sp1_runtime"),
                "sp2_runtime": rec.get("sp2_runtime"),
                "total_iteration_time": rec.get("total_iteration_time"),
                "total_time": rec.get("total_time"),
                "mp_mode": rec.get("mp_mode"),
                "lb_updated": rec.get("lb_updated"),
                "ub_updated": rec.get("ub_updated"),
                "new_pattern_added": rec.get("new_pattern_added"),
                "duplicate_pattern": rec.get("duplicate_pattern"),
                "sp2_feasible": rec.get("sp2_feasible"),
                "source_file": str(path),
            })

    run_df = pd.DataFrame(runs)
    for col in RUN_NUMERIC_COLS:
        run_df[col] = pd.to_numeric(run_df[col], errors="coerce")

    iter_df = pd.DataFrame(iterations)
    if not iter_df.empty:
        for col in ITER_NUMERIC_COLS:
            iter_df[col] = pd.to_numeric(iter_df[col], errors="coerce")
        iter_df = iter_df.sort_values(
            ["instance", "configuration", "iteration"], kind="stable"
        ).reset_index(drop=True)

    return run_df, iter_df


def validate_experiment(runs: pd.DataFrame, iterations: pd.DataFrame) -> None:
    """Reject missing/duplicate runs and warn about incomplete histories."""
    run_keys = ["instance", "configuration"]
    if runs.duplicated(run_keys).any():
        raise ValueError("Duplicate instance/configuration pairs exist.")
    if len(runs) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, found {len(runs)}.")
    if runs["instance"].nunique() != EXPECTED_INSTANCES:
        raise ValueError("Unexpected number of unique instances.")
    if runs["configuration"].nunique() != EXPECTED_CONFIGURATIONS:
        raise ValueError("Unexpected number of unique configurations.")
    counts = runs.groupby("configuration")["instance"].nunique()
    if not counts.eq(EXPECTED_INSTANCES).all():
        raise ValueError("Some configurations were not run on all instances.")
    if not runs["threads"].eq(EXPECTED_THREADS).all():
        raise ValueError(f"Expected exactly {EXPECTED_THREADS} threads in each run.")

    if not iterations.empty and iterations.duplicated(
        ["instance", "configuration", "iteration"]
    ).any():
        raise ValueError("Duplicate iteration records detected.")

    incomplete = runs.loc[
        runs["iterations"].gt(runs["history_records"])
    ]
    if not incomplete.empty:
        logging.warning(
            "%d runs have fewer recorded iterations than started iterations. "
            "Their final MP solve may be absent from the iteration history.",
            len(incomplete),
        )

    incompatible = runs.loc[
        runs["history_records"].ne(runs["completed_iterations"])
    ]
    if not incompatible.empty:
        logging.warning(
            "%d runs disagree between result.completed_iterations and "
            "the length of iteration_history.", len(incompatible)
        )


def augment_runs(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["absolute_gap"] = runs["ub"] - runs["lb"]
    runs["numerical_issue"] = (
        runs["status"].eq("NUMERICAL_BOUND_INCONSISTENCY")
        | runs["absolute_gap"].lt(-1e-8)
    )
    runs["proven"] = runs["status"].eq("OPTIMAL_PROVEN") & ~runs["numerical_issue"]
    runs["feasible"] = runs["has_feasible_solution"] & runs["ub"].notna()

    safe_denominator = runs["ub"].abs().where(runs["ub"].abs().gt(1e-9))
    runs["relative_gap_pct"] = (100.0 * runs["absolute_gap"] / safe_denominator).where(
        ~runs["numerical_issue"]
    )
    runs["par10_s"] = np.where(
        runs["proven"], runs["runtime_s"], 10.0 * runs["time_limit_s"]
    )
    runs["valid_ub"] = runs["ub"].where(runs["feasible"] & ~runs["numerical_issue"])
    runs["best_known_ub"] = runs.groupby("instance")["valid_ub"].transform("min")
    runs["ub_deviation_pct"] = (
        100.0 * (runs["valid_ub"] - runs["best_known_ub"])
        / np.maximum(1.0, runs["best_known_ub"].abs())
    )
    runs["mp_runtime_share"] = runs["mp_runtime_s"] / runs["runtime_s"].where(
        runs["runtime_s"].gt(0)
    )
    runs["unrecorded_iterations"] = (
        runs["iterations"] - runs["history_records"]
    )
    # Includes initialization, callbacks/diagnostics and any unrecorded partial iteration.
    runs["unaccounted_wall_s"] = (
        runs["runtime_s"] - runs["recorded_iteration_wall_s"]
    )
    # Reference is descriptive only. It is not proof of validity of a Big-M bound.
    runs["reference_final_lb"] = runs["lb"].where(~runs["numerical_issue"])
    runs["best_final_lb_instance"] = runs.groupby("instance")["reference_final_lb"].transform("max")
    runs["final_lb_deficit_pct"] = (
        100.0 * (runs["best_final_lb_instance"] - runs["lb"])
        / np.maximum(1.0, runs["best_final_lb_instance"].abs())
    )
    return add_status_metrics(runs)


def augment_iterations(iters: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if iters.empty:
        return iters

    iters = iters.copy()
    grouped = iters.groupby(["instance", "configuration"], sort=False)
    iters["mp_mip_gap_pct"] = 100.0 * iters["mp_mip_gap"]
    iters["mp_local_abs_gap"] = iters["mp_obj"] - iters["mp_bound"]
    iters["lb_improvement"] = grouped["lb"].diff()
    iters["cumulative_mp_runtime_s"] = grouped["mp_runtime"].cumsum()
    iters["cumulative_sp1_runtime_s"] = grouped["sp1_runtime"].cumsum()
    iters["cumulative_sp2_runtime_s"] = grouped["sp2_runtime"].cumsum()
    iters["mp_time_fraction"] = (
        iters["cumulative_mp_runtime_s"]
        / iters["total_time"].where(iters["total_time"].gt(0))
    )

    references = runs[[
        "instance", "configuration", "lb", "ub", "runtime_s",
        "best_final_lb_instance", "status",
    ]].rename(columns={
        "lb": "final_lb", "ub": "final_ub", "status": "final_status",
        "runtime_s": "final_runtime_s",
    })
    iters = iters.merge(
        references, on=["instance", "configuration"], validate="many_to_one"
    )
    iters["lb_distance_to_own_final"] = iters["final_lb"] - iters["lb"]
    iters["mp_bound_distance_to_own_final"] = iters["final_lb"] - iters["mp_bound"]
    # Signed diagnostic, NOT an MP MIP gap and not a bilevel optimality gap.
    iters["mp_obj_minus_final_lb"] = iters["mp_obj"] - iters["final_lb"]
    iters["lb_deficit_to_best_pct"] = (
        100.0 * (iters["best_final_lb_instance"] - iters["lb"])
        / np.maximum(1.0, iters["best_final_lb_instance"].abs())
    )
    iters["bilevel_absolute_gap"] = iters["ub"] - iters["lb"]
    iters["bilevel_relative_gap_pct"] = (
        100.0 * iters["bilevel_absolute_gap"]
        / iters["ub"].abs().where(iters["ub"].abs().gt(1e-9))
    )
    return iters


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.groupby(["formulation", "configuration"], as_index=False).agg(
        n_instances=("instance", "nunique"),
        proven=("proven", "sum"),
        feasible=("feasible", "sum"),
        numerical_issues=("numerical_issue", "sum"),
        median_runtime_s=("runtime_s", "median"),
        p90_runtime_s=("runtime_s", lambda s: s.quantile(0.90)),
        mean_par10_s=("par10_s", "mean"),
        median_gap_pct=("relative_gap_pct", "median"),
        median_ub_deviation_pct=("ub_deviation_pct", "median"),
        median_final_lb_deficit_pct=("final_lb_deficit_pct", "median"),
        median_mp_share=("mp_runtime_share", "median"),
        median_iterations=("iterations", "median"),
        incomplete_history_runs=("unrecorded_iterations", lambda s: s.gt(0).sum()),
        median_unaccounted_wall_s=("unaccounted_wall_s", "median"),
        **status_aggregations(),
    )
    return result.sort_values(
        ["formulation", "numerical_issues", "any_feasible", "proven",
         "median_unproven_feasible_gap_pct", "median_feasible_runtime_s"],
        ascending=[True, True, False, False, True, True],
    )


def summarize_mp(runs: pd.DataFrame, iters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run diagnostics and per-configuration averages of those diagnostics."""
    if iters.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["instance", "configuration"]
    first = iters.groupby(keys, sort=False).head(1).copy()
    last = iters.groupby(keys, sort=False).tail(1).copy()
    stats = iters.groupby(keys, as_index=False).agg(
        recorded_iterations=("iteration", "size"),
        polishing_iterations=("mp_mode", lambda s: s.eq("BOUND POLISHING").sum()),
        lb_updated_iterations=("lb_updated", lambda s: s.fillna(False).sum()),
        duplicate_pattern_iterations=("duplicate_pattern", lambda s: s.fillna(False).sum()),
        new_pattern_iterations=("new_pattern_added", lambda s: s.fillna(False).sum()),
        median_mp_mip_gap_pct=("mp_mip_gap_pct", "median"),
        max_mp_runtime_s=("mp_runtime", "max"),
        mean_mp_runtime_s=("mp_runtime", "mean"),
    )

    selected_first = first[keys + ["lb", "mp_mip_gap_pct", "mp_obj", "mp_bound"]].rename(
        columns={
            "lb": "first_lb", "mp_mip_gap_pct": "first_mp_gap_pct",
            "mp_obj": "first_mp_obj", "mp_bound": "first_mp_bound",
        }
    )
    selected_last = last[keys + ["lb", "mp_mip_gap_pct", "mp_obj", "mp_bound"]].rename(
        columns={
            "lb": "last_recorded_lb", "mp_mip_gap_pct": "last_mp_gap_pct",
            "mp_obj": "last_mp_obj", "mp_bound": "last_mp_bound",
        }
    )
    per_run = runs.merge(stats, on=keys, how="left", validate="one_to_one")
    per_run = per_run.merge(selected_first, on=keys, how="left", validate="one_to_one")
    per_run = per_run.merge(selected_last, on=keys, how="left", validate="one_to_one")
    per_run["lb_gain_from_first"] = per_run["lb"] - per_run["first_lb"]
    per_run["final_lb_minus_last_recorded"] = per_run["lb"] - per_run["last_recorded_lb"]
    per_run["mp_time_per_recorded_iteration_s"] = (
        per_run["mp_runtime_s"] / per_run["recorded_iterations"]
    )

    by_configuration = per_run.groupby(
        ["formulation", "configuration"], as_index=False
    ).agg(
        runs=("instance", "nunique"),
        median_mp_time_s=("mp_runtime_s", "median"),
        median_mp_fraction=("mp_runtime_share", "median"),
        median_mp_runtime_per_iteration_s=("mp_time_per_recorded_iteration_s", "median"),
        median_final_mp_gap_pct=("last_mp_gap_pct", "median"),
        median_lb_gain=("lb_gain_from_first", "median"),
        median_unaccounted_wall_s=("unaccounted_wall_s", "median"),
        median_final_lb_deficit_pct=("final_lb_deficit_pct", "median"),
        total_polishing_iterations=("polishing_iterations", "sum"),
        total_duplicates=("duplicate_pattern_iterations", "sum"),
        total_new_patterns=("new_pattern_iterations", "sum"),
        incomplete_history_runs=("unrecorded_iterations", lambda s: s.gt(0).sum()),
    )
    return per_run, by_configuration


def summarize_mp_modes(iters: pd.DataFrame) -> pd.DataFrame:
    """Contrast exploration with polishing across recorded MP iterations."""
    if iters.empty:
        return pd.DataFrame()
    return iters.groupby(
        ["formulation", "configuration", "mp_mode"], dropna=False, as_index=False
    ).agg(
        recorded_mp_solves=("iteration", "size"),
        median_mp_runtime_s=("mp_runtime", "median"),
        median_mp_gap_pct=("mp_mip_gap_pct", "median"),
        fraction_lb_improved=("lb_improvement", lambda s: s.gt(1e-9).mean()),
        total_lb_improvement_excl_first=("lb_improvement", lambda s: s.clip(lower=0).sum()),
    )


def save_tables(study_dir: Path, runs: pd.DataFrame, iters: pd.DataFrame) -> Path:
    output_dir = study_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs.to_csv(output_dir / "calibration_all_runs.csv", index=False)
    if not iters.empty:
        iters.to_csv(output_dir / "mp_iteration_history.csv", index=False)

    run_summary = summarize_runs(runs)
    run_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    save_status_tables(runs, output_dir)

    per_run, mp_summary = summarize_mp(runs, iters)
    if not per_run.empty:
        per_run.to_csv(output_dir / "mp_run_diagnostics.csv", index=False)
        mp_summary.to_csv(output_dir / "mp_configuration_summary.csv", index=False)

    modes = summarize_mp_modes(iters)
    if not modes.empty:
        modes.to_csv(output_dir / "mp_mode_comparison.csv", index=False)

    by_size = runs.groupby(
        ["formulation", "configuration", "size"], as_index=False
    ).agg(
        instances=("instance", "nunique"),
        proven=("proven", "sum"),
        **status_aggregations(),
        median_runtime_s=("runtime_s", "median"),
        mean_par10_s=("par10_s", "mean"),
        median_final_lb_deficit_pct=("final_lb_deficit_pct", "median"),
    )
    by_size.to_csv(output_dir / "calibration_by_size.csv", index=False)

    by_regime = runs.groupby(
        ["formulation", "configuration", "regime"], as_index=False
    ).agg(
        instances=("instance", "nunique"),
        proven=("proven", "sum"),
        **status_aggregations(),
        median_runtime_s=("runtime_s", "median"),
        mean_par10_s=("par10_s", "mean"),
        median_final_lb_deficit_pct=("final_lb_deficit_pct", "median"),
    )
    by_regime.to_csv(output_dir / "calibration_by_regime.csv", index=False)

    for measure, basename in (
        ("runtime_s", "calibration_runtime_comparison.csv"),
        ("relative_gap_pct", "calibration_gap_comparison.csv"),
        ("proven", "calibration_proven_comparison.csv"),
        ("lb", "calibration_final_lb_comparison.csv"),
    ):
        runs.pivot(index=["size", "regime", "instance"],
                   columns="configuration", values=measure).to_csv(output_dir / basename)

    return output_dir


def _prepare_axes(title: str, x_label: str, y_label: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    return fig, ax


def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_instance_comparisons(iters: pd.DataFrame, runs: pd.DataFrame, out_dir: Path) -> None:
    """For each of nine instances, compare SOS1 and Big-M SEPARATELY."""
    if iters.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for (instance, formulation), group in iters.groupby(["instance", "formulation"]):
        run_data = runs.loc[runs["instance"].eq(instance)]
        if run_data.empty:
            continue
        reference_lb = run_data["best_final_lb_instance"].max()
        safe_name = f"{instance}_{formulation}".replace("/", "-")

        # Plot A: global decomposition LB after each completed iteration
        fig, ax = _prepare_axes(
            f"Global lower-bound progression — {instance} ({formulation})",
            "Completed decomposition iteration", "Global lower bound",
        )
        for config_name, traces in group.groupby("configuration"):
            traces = traces.sort_values("iteration")
            ax.plot(traces["iteration"], traces["lb"], marker="o", markersize=3,
                    label=config_name)
        if pd.notna(reference_lb):
            ax.axhline(reference_lb, linestyle="--", linewidth=1.2,
                       label="Best final LB (all configurations)")
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, out_dir / f"lb_progress_{safe_name}.png")

        # Plot B: MIP gap of the MP at each completed iteration
        fig, ax = _prepare_axes(
            f"Master-problem MIP gap — {instance} ({formulation})",
            "Completed decomposition iteration", "MP Gurobi MIP gap (%)",
        )
        for config_name, traces in group.groupby("configuration"):
            traces = traces.sort_values("iteration")
            ax.plot(traces["iteration"], traces["mp_mip_gap_pct"], marker="o",
                    markersize=3, label=config_name)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, out_dir / f"mp_gap_{safe_name}.png")

        # Plot C: LB progression by elapsed decomposition wall-clock time
        fig, ax = _prepare_axes(
            f"Lower-bound progression vs runtime — {instance} ({formulation})",
            "Elapsed wall-clock time after completed iteration (s)",
            "Global lower bound",
        )
        for config_name, traces in group.groupby("configuration"):
            traces = traces.sort_values("iteration")
            ax.plot(traces["total_time"], traces["lb"], marker="o", markersize=3,
                    label=config_name)
        if pd.notna(reference_lb):
            ax.axhline(reference_lb, linestyle="--", linewidth=1.2,
                       label="Best final LB (all configurations)")
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, out_dir / f"lb_vs_time_{safe_name}.png")


def plot_detailed_runs(iters: pd.DataFrame, out_dir: Path) -> None:
    """Optional: plot MP incumbent, MP dual bound and global LB for every run."""
    if iters.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for (instance, config), group in iters.groupby(["instance", "configuration"]):
        group = group.sort_values("iteration")
        fig, ax = _prepare_axes(
            f"MP incumbent, MP bound, global LB — {instance}\n{config}",
            "Completed decomposition iteration", "Objective value",
        )
        ax.plot(group["iteration"], group["mp_obj"], marker="o",
                label="MP incumbent (local)")
        ax.plot(group["iteration"], group["mp_bound"], marker="o",
                label="MP dual bound (local)")
        ax.plot(group["iteration"], group["lb"], marker="o",
                label="Global decomposition LB")
        ref = group["final_lb"].iloc[0]
        if pd.notna(ref):
            ax.axhline(ref, linestyle="--", label="Own final global LB")
        ax.legend(fontsize=8)
        safe_name = f"{instance}_{config}".replace("/", "-")
        _save_plot(fig, out_dir / f"mp_details_{safe_name}.png")

# =============================================================================
# Bilevel convergence analysis: complementary to the existing MP figures
# =============================================================================

def build_bilevel_iteration_table(iters: pd.DataFrame) -> pd.DataFrame:
    """Export recorded post-iteration bounds and their progress.

    A row is an observation AFTER a completed iteration. A missing UB means
    no bilevel-feasible upper bound was available at that checkpoint.
    """
    columns = [
        "instance", "size", "regime", "configuration", "formulation",
        "iteration", "total_time", "lb", "ub", "bilevel_absolute_gap",
        "bilevel_relative_gap_pct", "lb_updated", "ub_updated", "sp2_feasible",
        "final_lb", "final_ub", "final_runtime_s", "final_status",
    ]
    if iters.empty:
        return pd.DataFrame(columns=columns + [
            "finite_lb", "finite_ub", "finite_gap", "lb_gain_since_previous",
            "ub_reduction_since_previous", "gap_reduction_since_previous",
        ])

    table = iters[columns].copy()
    for key in ("lb", "ub", "bilevel_absolute_gap"):
        table[key] = pd.to_numeric(table[key], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    table["finite_lb"] = np.isfinite(table["lb"])
    table["finite_ub"] = np.isfinite(table["ub"])
    table["finite_gap"] = np.isfinite(table["bilevel_absolute_gap"])
    group = table.groupby(["instance", "configuration"], sort=False)
    table["lb_gain_since_previous"] = group["lb"].diff()
    table["ub_reduction_since_previous"] = -group["ub"].diff()
    table["gap_reduction_since_previous"] = -group["bilevel_absolute_gap"].diff()
    return table


def build_bilevel_run_diagnostics(
    runs: pd.DataFrame, bilevel: pd.DataFrame
) -> pd.DataFrame:
    """Run-level metrics, including first *recorded* finite UB checkpoint."""
    keys = ["instance", "configuration"]
    extra = [
        "first_recorded_finite_ub_iteration", "first_recorded_finite_ub_time_s",
        "last_recorded_lb", "last_recorded_ub", "last_recorded_gap",
        "last_recorded_time_s",
    ]
    if bilevel.empty:
        out = runs.copy()
        for field in extra:
            out[field] = np.nan
        return out
    
    ordered = bilevel.sort_values(keys + ["iteration"])
    first_feasible = ordered.loc[ordered["finite_ub"]].groupby(keys).head(1)
    first_feasible = first_feasible[keys + ["iteration", "total_time"]].rename(
        columns={
            "iteration": "first_recorded_finite_ub_iteration",
            "total_time": "first_recorded_finite_ub_time_s",
        }
    )
    last = ordered.groupby(keys).tail(1)[
        keys + ["lb", "ub", "bilevel_absolute_gap", "total_time"]
    ].rename(columns={
        "lb": "last_recorded_lb",
        "ub": "last_recorded_ub",
        "bilevel_absolute_gap": "last_recorded_gap",
        "total_time": "last_recorded_time_s",
    })
    out = runs.merge(first_feasible, on=keys, how="left", validate="one_to_one")
    out = out.merge(last, on=keys, how="left", validate="one_to_one")
    out["final_gap"] = out["ub"] - out["lb"]
    out["final_lb_minus_last_recorded"] = out["lb"] - out["last_recorded_lb"]
    out["final_ub_minus_last_recorded"] = out["ub"] - out["last_recorded_ub"]
    out["last_checkpoint_to_termination_s"] = (
        out["runtime_s"] - out["last_recorded_time_s"]
    )
    return out


def plot_bilevel_comparisons(
    bilevel: pd.DataFrame, runs: pd.DataFrame, out_dir: Path
) -> None:
    """Four bilevel plots per instance/formulation, keeping MP plots unchanged.

    For all iteration and time curves, observations are END-OF-COMPLETED-ITERATION
    snapshots. The latest known bound is shown as a step function, not an
    interpolated within-solve trajectory. No artificial UB is used before a
    bilevel-feasible solution is recorded.

    For TIME plots only, an unfilled diamond represents the reported terminal
    bounds at total run wall-clock time, which may be later than the last
    completed iteration. Diamonds are not connected by lines, since the time
    of any unrecorded bound change is unknown.
    """
    if bilevel.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = (
        ("iteration", "Completed decomposition iteration", "iterations"),
        ("total_time", "Elapsed wall-clock time after completed iteration (s)", "time"),
    )

    for (instance, formulation), group in bilevel.groupby(
        ["instance", "formulation"], sort=True
    ):
        run_group = runs.loc[
            runs["instance"].eq(instance) & runs["formulation"].eq(formulation)
        ].set_index("configuration")
        config_names = sorted(group["configuration"].unique())
        # Use identical configuration colors across all four figures.
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {name: cycle[i % len(cycle)] for i, name in enumerate(config_names)}
        safe_name = f"{instance}_{formulation}".replace("/", "-")

        for xkey, xlabel, suffix in styles:
            is_time = xkey == "total_time"

            # -------------------------------------------------------------
            # Plot 1: two bounds on a common objective-value axis.
            # -------------------------------------------------------------
            fig, ax = _prepare_axes(
                f"Bilevel bounds vs {suffix} — {instance} ({formulation})",
                xlabel, "Leader objective value (global LB / UB)",
            )
            for name in config_names:
                records = group.loc[group["configuration"].eq(name)].sort_values("iteration")
                color = colors[name]
                ax.step(records[xkey], records["lb"], where="post", color=color,
                        linestyle="-", marker="o", markersize=2.5, linewidth=1.2,
                        label=name)
                finite_ub = records["ub"].where(np.isfinite(records["ub"]))
                ax.step(records[xkey], finite_ub, where="post", color=color,
                        linestyle="--", marker=".", markersize=3, linewidth=1.2)
                if is_time and name in run_group.index:
                    run = run_group.loc[name]
                    for value in (run["lb"], run["ub"]):
                        if pd.notna(value) and np.isfinite(value):
                            ax.scatter([run["runtime_s"]], [value], marker="D",
                                       facecolors="none", edgecolors=color,
                                       s=34, linewidths=1.1, zorder=5)
            ax.legend(
                title="Config colors; LB solid, UB dashed",
                fontsize=7, title_fontsize=8, loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
            if is_time:
                ax.text(0.0, -0.13, "Hollow diamonds: reported terminal LB / UB",
                        fontsize=8, transform=ax.transAxes)
            _save_plot(fig, out_dir / f"bilevel_bounds_vs_{suffix}_{safe_name}.png")

            # -------------------------------------------------------------
            # Plot 2: global bilevel absolute optimality gap (UB - LB).
            # -------------------------------------------------------------
            fig, ax = _prepare_axes(
                f"Bilevel optimality gap vs {suffix} — {instance} ({formulation})",
                xlabel, "Global absolute optimality gap (UB − LB)",
            )
            for name in config_names:
                records = group.loc[group["configuration"].eq(name)].sort_values("iteration")
                color = colors[name]
                # Never display undefined gaps as zero, or numerically invalid
                # negative gaps as progress.
                gap = records["bilevel_absolute_gap"].where(
                    records["bilevel_absolute_gap"].ge(-1e-8)
                ).clip(lower=0)
                ax.step(records[xkey], gap, where="post", color=color,
                        marker="o", markersize=2.5, linewidth=1.2, label=name)
                if is_time and name in run_group.index:
                    run = run_group.loc[name]
                    if (
                        pd.notna(run["lb"]) and pd.notna(run["ub"])
                        and np.isfinite(run["lb"]) and np.isfinite(run["ub"])
                    ):
                        terminal_gap = run["ub"] - run["lb"]
                        if terminal_gap >= -1e-8:
                            ax.scatter([run["runtime_s"]], [max(terminal_gap, 0)],
                                       marker="D", facecolors="none",
                                       edgecolors=color, s=34, linewidths=1.1,
                                       zorder=5)
            ax.set_ylim(bottom=0)
            ax.legend(
                fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            )
            if is_time:
                ax.text(0.0, -0.13, "Hollow diamonds: reported terminal UB − LB",
                        fontsize=8, transform=ax.transAxes)
            _save_plot(fig, out_dir / f"bilevel_gap_vs_{suffix}_{safe_name}.png")


def main(study_dir: Path, generate_plots: bool, all_run_plots: bool, bilevel_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    runs, iterations = load_results(study_dir)
    validate_experiment(runs, iterations)
    runs = augment_runs(runs)
    iterations = augment_iterations(iterations, runs)
    output_dir = save_tables(study_dir, runs, iterations)

    # Supplement MP tables with explicit bilevel bound/gap checkpoints.
    bilevel = build_bilevel_iteration_table(iterations)
    bilevel.to_csv(output_dir / "bilevel_iteration_history.csv", index=False)
    build_bilevel_run_diagnostics(runs, bilevel).to_csv(
        output_dir / "bilevel_run_diagnostics.csv", index=False
    )

    # Existing MP plots are unchanged and remain the default.
    if generate_plots and not bilevel_only:
        plot_instance_comparisons(iterations, runs, output_dir / "figures" / "by_instance")
    if generate_plots:
        plot_bilevel_comparisons(
            bilevel, runs, output_dir / "figures" / "bilevel_by_instance"
        )
    if all_run_plots and not bilevel_only:
        plot_detailed_runs(iterations, output_dir / "figures" / "all_runs")

    print(f"Loaded {len(runs)} runs and {len(iterations)} recorded iterations.")
    print(f"Outputs: {output_dir}")
    for formulation, part in summarize_runs(runs).groupby("formulation"):
        print(f"\n{formulation} configurations:")
        cols = [
            "configuration", "n_proven", "n_incumbent_match",
            "n_feasible_suboptimal", "n_mp_no_solution", "n_no_bilevel_feasible",
            "any_feasible", "median_unproven_feasible_gap_pct",
            "median_feasible_runtime_s",
        ]
        print(part[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--all-run-plots", action="store_true")
    parser.add_argument(
        "--bilevel-only", action="store_true",
        help="Create only the new bilevel figures; do not regenerate MP figures.",
    )
    args = parser.parse_args()
    main(args.study_dir, generate_plots=not args.no_plots,
         all_run_plots=args.all_run_plots, bilevel_only=args.bilevel_only)