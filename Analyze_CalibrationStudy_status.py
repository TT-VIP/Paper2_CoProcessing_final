
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STUDY_DIR = ROOT / "Solutions" / "calib"



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


def read_calibration_result(path: Path) -> dict:
    """Extract relevant algorithm performance metrics from one solution JSON."""

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    metadata = data["metadata"]
    config = data["algorithm_config"]
    result = data["result"]
    history = data.get("iteration_history", [])

    def cumulative_runtime(key: str) -> float:
        return sum(
            float(record.get(key) or 0.0)
            for record in history
        )

    return {
        # Instance identification
        "instance": metadata["instance_name"],
        "size": metadata["size_class"],
        "regime": metadata["structural_regime"],

        # Algorithm configuration
        "configuration": metadata["method_tag"],
        "formulation": config["complementarity_formulation"],
        "strengthening": config["primal_dual_strengthening"],
        "big_m": config["bigM_duals_unrestricted"],
        "cutoff_tolerance": config["cutoff_bound_tolerance"],
        "threads": config.get("threads"),

        # Solver result
        "status": result["status"],
        "termination_reason": result["termination_reason"],
        "runtime_s": result["total_solution_time"],
        "time_limit_s": config["total_time_limit"],
        "iterations": result["iterations"],
        "lb": result["lower_bound"],
        "ub": result["upper_bound"],
        "has_feasible_solution": data["best_solution"] is not None,

        # Cumulative runtimes of recorded iterations
        "mp_runtime_s": cumulative_runtime("mp_runtime"),
        "sp1_runtime_s": cumulative_runtime("sp1_runtime"),
        "sp2_runtime_s": cumulative_runtime("sp2_runtime"),

        # Original source
        "source_file": str(path),
    }


# -------------------------------------------------------------------------
# Load all solutions
# -------------------------------------------------------------------------

json_files = sorted(STUDY_DIR.rglob("SOL_*.json"))

if not json_files:
    raise FileNotFoundError(
        f"No calibration solution JSON files found in {STUDY_DIR}"
    )

df = pd.DataFrame(
    read_calibration_result(path)
    for path in json_files
)

# Ensure numeric columns have numeric types
numeric_columns = [
    "big_m",
    "cutoff_tolerance",
    "threads",
    "runtime_s",
    "time_limit_s",
    "iterations",
    "lb",
    "ub",
    "mp_runtime_s",
    "sp1_runtime_s",
    "sp2_runtime_s",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# -------------------------------------------------------------------------
# Validate experimental design
# -------------------------------------------------------------------------

if df.duplicated(["instance", "configuration"]).any():
    raise ValueError(
        "Duplicate instance/configuration combinations found."
    )

if len(df) != 135:
    raise ValueError(
        f"Expected 135 calibration results, found {len(df)}."
    )

if df["instance"].nunique() != 9:
    raise ValueError("Expected 9 distinct instances.")

if df["configuration"].nunique() != 15:
    raise ValueError("Expected 15 distinct configurations.")

if not df.groupby("configuration")["instance"].nunique().eq(9).all():
    raise ValueError(
        "Not every configuration has results for all 9 instances."
    )

if not df["threads"].eq(16).all():
    raise ValueError(
        "Not all configurations used the intended 16 threads."
    )

print("Calibration results loaded and validated.")
print(f"Results: {len(df)}")
print(f"Instances: {df['instance'].nunique()}")
print(f"Configurations: {df['configuration'].nunique()}")



# -------------------------------------------------------------------------
# Convergence and numerical validity
# -------------------------------------------------------------------------

df["absolute_gap"] = df["ub"] - df["lb"]

df["numerical_issue"] = (
    df["status"].eq("NUMERICAL_BOUND_INCONSISTENCY")
    | df["absolute_gap"].lt(-1e-8)
)

df["proven"] = (
    df["status"].eq("OPTIMAL_PROVEN")
    & ~df["numerical_issue"]
)

df["feasible"] = (
    df["has_feasible_solution"]
    & df["ub"].notna()
)

# Relative final gap, only where mathematically meaningful
df["relative_gap_pct"] = (
    100.0 * df["absolute_gap"] / df["ub"].abs()
)

invalid_gap = (
    df["absolute_gap"].isna()
    | df["numerical_issue"]
    | df["ub"].abs().le(1e-9)
)

df.loc[invalid_gap, "relative_gap_pct"] = np.nan

# -------------------------------------------------------------------------
# PAR10 for PROVEN OPTIMALITY only; retain for comparison but do not rank
# configurations on PAR10 alone when large instances seldom prove optimality.
# -------------------------------------------------------------------------

df["par10_s"] = np.where(
    df["proven"],
    df["runtime_s"],
    10.0 * df["time_limit_s"],
)

# -------------------------------------------------------------------------
# Incumbent comparison within the SAME instance
# -------------------------------------------------------------------------

df["valid_ub"] = df["ub"].where(
    df["feasible"] & ~df["numerical_issue"]
)

df["best_known_ub"] = (
    df.groupby("instance")["valid_ub"].transform("min")
)

# Percentage deterioration relative to the best found feasible solution
df["ub_deviation_pct"] = (
    100.0
    * (df["valid_ub"] - df["best_known_ub"])
    / np.maximum(1.0, df["best_known_ub"].abs())
)

# Approximate MP contribution to total wall-clock runtime
df["mp_runtime_share"] = (
    df["mp_runtime_s"] / df["runtime_s"]
)

df = add_status_metrics(df)

# -------------------------------------------------------------------------
# Export detailed results
# -------------------------------------------------------------------------

df.to_csv(
    STUDY_DIR / "calibration_all_runs.csv",
    index=False,
)



# -------------------------------------------------------------------------
# Configuration-level summary
# -------------------------------------------------------------------------

summary = (
    df.groupby(
        ["formulation", "configuration"],
        as_index=False,
    )
    .agg(
        n_instances=("instance", "nunique"),
        proven=("proven", "sum"),
        feasible=("feasible", "sum"),
        numerical_issues=("numerical_issue", "sum"),

        median_runtime_s=("runtime_s", "median"),
        p90_runtime_s=(
            "runtime_s",
            lambda x: x.quantile(0.9),
        ),
        mean_par10_s=("par10_s", "mean"),

        median_gap_pct=("relative_gap_pct", "median"),
        median_ub_deviation_pct=(
            "ub_deviation_pct",
            "median",
        ),

        median_mp_share=("mp_runtime_share", "median"),
        median_iterations=("iterations", "median"),
        **status_aggregations(),
    )
)

summary = summary.sort_values(
    by=[
        "formulation",
        "numerical_issues",
        "any_feasible",
        "proven",
        "median_unproven_feasible_gap_pct",
        "median_feasible_runtime_s",
    ],
    ascending=[True, True, False, False, True, True],
)

summary.to_csv(
    STUDY_DIR / "calibration_summary.csv",
    index=False,
)

save_status_tables(df, STUDY_DIR)

for formulation in ("SOS1", "Big-M"):
    print(f"\n{'=' * 70}")
    print(f"{formulation} CALIBRATION RESULTS")
    print("=" * 70)

    selection = summary.loc[
        summary["formulation"].eq(formulation),
        [
            "configuration",
            "n_proven",
            "n_incumbent_match",
            "n_feasible_suboptimal",
            "n_mp_no_solution",
            "n_no_bilevel_feasible",
            "any_feasible",
            "median_unproven_feasible_gap_pct",
            "median_feasible_runtime_s",
        ],
    ]

    print(selection.to_string(index=False, float_format="%.3f"))

gap_comparison = df.pivot(
    index=["size", "regime", "instance"],
    columns="configuration",
    values="relative_gap_pct",
)

gap_comparison.to_csv(
    STUDY_DIR / "calibration_gap_comparison.csv"
)

runtime_comparison = df.pivot(
    index=["size", "regime", "instance"],
    columns="configuration",
    values="runtime_s",
)

runtime_comparison.to_csv(
    STUDY_DIR / "calibration_runtime_comparison.csv"
)

proven_comparison = df.pivot(
    index=["size", "regime", "instance"],
    columns="configuration",
    values="proven",
)

proven_comparison.to_csv(
    STUDY_DIR / "calibration_proven_comparison.csv"
)