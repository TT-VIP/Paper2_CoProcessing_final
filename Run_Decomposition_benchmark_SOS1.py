import sys
from pathlib import Path
import logging
import math
import time

# Ensure project root is importable (so "Instances" resolves)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Instances.json_reader import read_instance_data_from_json, read_instance_metadata_from_json
from Yue_Decomposition_Algorithm.Decomposition_Algorithm import run_yue_decomposition
from Yue_Decomposition_Algorithm.Normalization import determine_normalization_bounds
from Yue_Decomposition_Algorithm.Bilevel_solution_export import build_solution_dictionary, write_solution_json


# =============================================================================
# Benchmark-study design
# =============================================================================

SEEDS = (7,19,37,61,89)

SIZE_CLASSES = {
    "small": "S",
    "medium": "M",
    "large": "L",
}

STRUCTURAL_REGIMES = (
    "balanced",
    "incDom",
    "cemDom",
)

# SOS1 benchmark results are stored separately from Big-M runs.
STUDY_OUTPUT_FOLDER = "SOS1"
INSTANCE_FOLDER = "Instances_CaseStudy_V3"

###############################################################################################
################################# Helper Functions ############################################
###############################################################################################
#region Helper functions
def setup_logger(
        *,
        instance_name: str, 
        instance_size_class: str, 
        instance_regime: str,
        method_tag: str,    
    ) -> tuple[Path, Path]:
    """
    Setup logging to file and console
    Creates a run folder named after the log file (without the .log extension)
    inside the solutions directory, and stores the log file inside that folder.
    """
    # Create solutions folder if it doesn't exist
    base_dir = ROOT / "Solutions" / STUDY_OUTPUT_FOLDER / instance_size_class / instance_regime
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an instance- and method-specific log filename
    log_filename = f"SOL_{instance_name}_{method_tag}"

    # Run folder has the same name as the log file (without extension)
    run_dir = base_dir / log_filename
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = run_dir / f"{log_filename}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        force=True,             # Ensure that the logging configuration is applied and installed new even if logging has been configured before
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_path, run_dir

def log_instance_metadata(metadata: dict) -> None:
    """Log instance metadata in a structured format"""
    logging.info("Instance Metadata:")
    for key, value in metadata.items():
        # logging.info(f"  {key}:")
        if isinstance(value, dict):
            logging.info(f"  {key}:")
            for nested_key, nested_value in value.items():    
                logging.info(f"    {nested_key}: {nested_value}")
        else:
            logging.info(f"  {key}: {value}")

    logging.info("-" * 40)

def log_run_metadata(
    *,
    mp_normal_time_limit: float,
    mp_polish_time_limit: float,
    sp1_max_time: float,
    sp2_max_time: float,
    threads: int,
    lb_stall_trigger: int,
    mip_gap: float,
    Xi: float,
    max_iterations: int,
    total_runtime: float,
    shutdown_buffer: float,
    weight_env: float,
    weight_mon: float,
    objective_scale: float,
    primal_dual_strenghtening: bool,
    bound_cutoff: bool,
    cutoff_bound_tolerance: float,
) -> None:
    """Log all settings defining the solver run."""
    logging.info("=" * 70)
    logging.info("SOLVER RUN CONFIGURATION")
    logging.info("=" * 70)

    logging.info(f"Xi: {Xi:g}")
    logging.info(f"Master MIP-gap (Gurobi): {mip_gap:g}")
    logging.info(f"Maximum decomposition iterations: {max_iterations}")
    logging.info(f"Total runtime limit: {total_runtime:g} s")
    logging.info(f"Shutdown buffer: {shutdown_buffer:g} s")
    logging.info(f"MP normal time limit: {mp_normal_time_limit:g} s")
    logging.info(f"MP polishing time limit: {mp_polish_time_limit:g} s")
    logging.info(f"LB stall trigger: {lb_stall_trigger}")
    logging.info(f"SP1 max time: {sp1_max_time:g} s")
    logging.info(f"SP2 max time: {sp2_max_time:g} s")
    logging.info(f"Threads per Gurobi solver call: {threads}")
    logging.info(f"Objective weights: environment={weight_env:g}, monetary={weight_mon:g}")
    logging.info(f"Objective scale: {objective_scale:g}")

    logging.info("Complementarity formulation: SOS1")
    logging.info(f"Primal-dual strengthening: {primal_dual_strenghtening}")

    logging.info(f"Bound cutoff: {'enabled' if bound_cutoff else 'disabled'}")
    logging.info(
        f"Bound cutoff tolerance: {cutoff_bound_tolerance:g}"
        if bound_cutoff else "Bound cutoff tolerance: N/A"
    )
    logging.info("=" * 70)

def format_power_of_ten(value: float) -> str:
    """Format positive powers of ten cleanly for filenames, e.g. 1e-5."""
    if value <= 0:
        raise ValueError("Cutoff tolerance must be positive.")

    exponent = int(round(math.log10(value)))

    if math.isclose(value, 10 ** exponent):
        return f"1e{exponent}"

    return f"{value:g}"

def get_method_tag(cutoff_bound_tolerance: float) -> str:
    """Tag the fixed SOS1 variant and its bound-cutoff tolerance."""
    return f"SOS1_cutTol{format_power_of_ten(cutoff_bound_tolerance)}"


def extract_normalization_bounds(instance) -> dict[str, float]:
    return {
        "emission_min": float(instance.Emission_min),
        "emission_max": float(instance.Emission_max),
        "cost_min": float(instance.Cost_min),
        "cost_max": float(instance.Cost_max),
    }


def apply_normalization_bounds(
    instance,
    bounds: dict[str, float],
) -> None:

    instance.Emission_min = bounds["emission_min"]
    instance.Emission_max = bounds["emission_max"]
    instance.Cost_min = bounds["cost_min"]
    instance.Cost_max = bounds["cost_max"]
#endregion

###############################################################################################
#################################### Run Algorithm ############################################
###############################################################################################
if __name__ == "__main__":
    # =========================================================================
    # Fixed algorithm settings
    #
    # Preserve the Big-M benchmark settings except for the SOS1 formulation
    # and the cutoff tolerance selected in the calibration study.
    # =========================================================================
    FIXED_ALGORITHM_CONFIG = {
        "Verbose": True,

        "mp_normal_time_limit": 200,
        "mp_polish_time_limit": 600,
        "sp1_max_time": 60,
        "sp2_max_time": 60,
        "threads": 16,

        "lb_stall_trigger": 2,
        "mip_gap": 1e-4,
        "Xi": 1e-4,

        "max_iterations": 100,

        "sos1_cuts": True,
        "primal_dual_strenghtening": False,
        "use_analytical_dual_bounds": False,  # Not applicable to SOS1.
        "cutoff_bound_tolerance": 1e-5,

        "total_time_limit": 3630,
        "shutdown_buffer": 30,

        "weight_env": 1.0,
        "weight_mon": 1.0,
        "objective_scale": 100.0,

        "bound_cutoff": True,
    }

    # =========================================================================
    # Solution-export settings
    # =========================================================================
    include_zeros = True
    zero_tolerance = 1e-8

    # =========================================================================
    # Generate experimental design
    # =========================================================================
    instance_specs = []

    for size_class, size_letter in SIZE_CLASSES.items():
        for regime in STRUCTURAL_REGIMES:
            for seed in SEEDS:
                instance_name = f"TI_{size_letter}_{regime}_S{seed:03d}.json"
                instance_file = (
                    ROOT / "Instances" / INSTANCE_FOLDER / size_class / regime / instance_name
                )

                instance_specs.append(
                    {
                        "size_class": size_class,
                        "size_letter": size_letter,
                        "regime": regime,
                        "seed": seed,
                        "instance_file": instance_file,
                    }
                )

    # =========================================================================
    # Print study design before starting expensive solves
    # =========================================================================
    n_instances = len(instance_specs)

    if n_instances != 45:
        raise RuntimeError(f"Expected 45 benchmark instances, got {n_instances}.")

    missing_files = [spec["instance_file"] for spec in instance_specs if not spec["instance_file"].is_file()]

    if missing_files:
        raise FileNotFoundError(
            "Missing benchmark instances:\n"
            + "\n".join(str(path) for path in missing_files)
        )

    print(f"Seeds:     {SEEDS}")
    print(f"Instances: {n_instances}")

    # =========================================================================
    # Overall Benchmark loop
    # =========================================================================
    for run_number, instance_spec in enumerate(instance_specs, start=1):
        instance_file = instance_spec["instance_file"]
        size_class = instance_spec["size_class"]
        regime = instance_spec["regime"]
        seed = instance_spec["seed"]

        # ---------------------------------------------------------------------
        # Run the algorithm configurations
        # ---------------------------------------------------------------------
        method_tag = get_method_tag(
            FIXED_ALGORITHM_CONFIG["cutoff_bound_tolerance"]
        )

        instance_name = instance_file.stem
        print("\n" + "=" * 80)
        print(f"RUN {run_number:03d}/{n_instances}: {instance_name} | {method_tag}")
        print("=" * 80)

        try:
            # ---------------------------------------------------------------------
            # Logging
            # ---------------------------------------------------------------------

            log_path, run_dir = setup_logger(
                instance_name=instance_name,
                instance_size_class=size_class,
                instance_regime=regime,
                method_tag=method_tag,
            )

            logging.info("=" * 80)
            logging.info("COMPUTATIONAL STUDY - BENCHMARK (SOS1)")
            logging.info("=" * 80)
            logging.info(f"Overall run: {run_number}/{n_instances}")
            logging.info(f"Instance file: {instance_file}")
            logging.info(f"Configuration: {method_tag}")
            logging.info(f"Output folder: {run_dir}")
            logging.info("=" * 80)

            instance_metadata = read_instance_metadata_from_json(instance_file)
            log_instance_metadata(instance_metadata)

            # -------------------------------------------------------------
            # Normalization: its messages and failures are now logged.
            # This step remains OUTSIDE the decomposition runtime budget.
            # -------------------------------------------------------------
            normalization_instance = read_instance_data_from_json(instance_file)

            logging.info(f"Computing normalization bounds for {instance_name} ...")
            normalization_start = time.perf_counter()
            determine_normalization_bounds(normalization_instance)
            normalization_runtime = time.perf_counter() - normalization_start
            normalization_bounds = extract_normalization_bounds(normalization_instance)

            logging.info("Normalization completed in %.2f s", normalization_runtime)
            for key, value in normalization_bounds.items():
                logging.info("  %s: %.10g", key, value)

            # Keep a fresh instance for decomposition, as in the original script.
            instance = read_instance_data_from_json(instance_file)
            apply_normalization_bounds(instance, normalization_bounds)

            log_run_metadata(
                mp_normal_time_limit=(FIXED_ALGORITHM_CONFIG["mp_normal_time_limit"]),
                mp_polish_time_limit=(FIXED_ALGORITHM_CONFIG["mp_polish_time_limit"]),
                sp1_max_time=(FIXED_ALGORITHM_CONFIG["sp1_max_time"]),
                sp2_max_time=(FIXED_ALGORITHM_CONFIG["sp2_max_time"]),
                threads=(FIXED_ALGORITHM_CONFIG["threads"]),
                lb_stall_trigger=(FIXED_ALGORITHM_CONFIG["lb_stall_trigger"]),
                mip_gap=(FIXED_ALGORITHM_CONFIG["mip_gap"]),
                Xi=FIXED_ALGORITHM_CONFIG["Xi"],
                max_iterations=(FIXED_ALGORITHM_CONFIG["max_iterations"]),
                total_runtime=(FIXED_ALGORITHM_CONFIG["total_time_limit"]),
                shutdown_buffer=(FIXED_ALGORITHM_CONFIG["shutdown_buffer"]),
                weight_env=(FIXED_ALGORITHM_CONFIG["weight_env"]),
                weight_mon=(FIXED_ALGORITHM_CONFIG["weight_mon"]),
                objective_scale=(FIXED_ALGORITHM_CONFIG["objective_scale"]),
                primal_dual_strenghtening=(FIXED_ALGORITHM_CONFIG["primal_dual_strenghtening"]),
                bound_cutoff=FIXED_ALGORITHM_CONFIG["bound_cutoff"],
                cutoff_bound_tolerance=(FIXED_ALGORITHM_CONFIG["cutoff_bound_tolerance"]),
            )

            # ---------------------------------------------------------------------
            # Decomposition
            # ---------------------------------------------------------------------
            logging.info(f"Starting Yue-KKT Decomposition Algorithm. Log: {log_path}")
            logging.info(f"SP2 IIS-files will be saved to {run_dir / 'SP2_IIS'}")
            
            decomposition_solution = run_yue_decomposition(
                **FIXED_ALGORITHM_CONFIG,
                instance=instance,
                solution_dir=run_dir,
            )

            algorithm_config = {
                "study": "Benchmark_SOS1",
                "instance_version": INSTANCE_FOLDER,
                "seed": seed,
                "configuration_id": method_tag,

                # Decomposition settings
                # ---------------------------------------------------------
                "xi": FIXED_ALGORITHM_CONFIG["Xi"],
                "max_iterations": FIXED_ALGORITHM_CONFIG["max_iterations"],
                "total_time_limit": FIXED_ALGORITHM_CONFIG["total_time_limit"],
                "shutdown_buffer": FIXED_ALGORITHM_CONFIG["shutdown_buffer"],

                # Solver settings
                # ---------------------------------------------------------
                "threads": FIXED_ALGORITHM_CONFIG["threads"],
                "mip_gap": FIXED_ALGORITHM_CONFIG["mip_gap"],
                "mp_normal_time_limit": FIXED_ALGORITHM_CONFIG["mp_normal_time_limit"],
                "mp_polish_time_limit": FIXED_ALGORITHM_CONFIG["mp_polish_time_limit"],
                "sp1_max_time": FIXED_ALGORITHM_CONFIG["sp1_max_time"],
                "sp2_max_time": FIXED_ALGORITHM_CONFIG["sp2_max_time"],
                "lb_stall_trigger": FIXED_ALGORITHM_CONFIG["lb_stall_trigger"],

                # Leader objective
                # ---------------------------------------------------------
                "weight_env": FIXED_ALGORITHM_CONFIG["weight_env"],
                "weight_mon": FIXED_ALGORITHM_CONFIG["weight_mon"],
                "objective_scale": FIXED_ALGORITHM_CONFIG["objective_scale"],

                # Reformulation
                # ---------------------------------------------------------
                "complementarity_formulation": ("SOS1" if FIXED_ALGORITHM_CONFIG["sos1_cuts"] else "Big-M"),
                "sos1_cuts": FIXED_ALGORITHM_CONFIG["sos1_cuts"],
                "primal_dual_strengthening": (FIXED_ALGORITHM_CONFIG["primal_dual_strenghtening"] if FIXED_ALGORITHM_CONFIG["sos1_cuts"] else None),
                # Retain the same export schema for comparisons with Big-M.
                "bigM_duals_unrestricted": None,
                "use_analytical_dual_bounds": False,
                "analytical_dual_bound_margin": None,
                "effective_dual_bigM": None,

                # Bound cutoff
                # ---------------------------------------------------------
                "bound_cutoff": FIXED_ALGORITHM_CONFIG["bound_cutoff"],
                "cutoff_bound_tolerance": FIXED_ALGORITHM_CONFIG["cutoff_bound_tolerance"],

                # Objective normalization
                # ---------------------------------------------------------
                "normalization_bounds": normalization_bounds.copy(),
            }

            # ---------------------------------------------------------------------
            # JSON solution export
            # ---------------------------------------------------------------------

            solution_dict = build_solution_dictionary(
                decomp_sol=decomposition_solution,
                instance=instance,
                method_tag=method_tag,
                algorithm_config=algorithm_config,
                include_zeros=include_zeros,
                zero_tolerance=zero_tolerance,
            )

            json_path = log_path.with_suffix(".json")

            write_solution_json(
                solution=solution_dict,
                output_path=json_path,
            )

            logging.info("")
            logging.info(f"Machine-readable solution written to {json_path}")

        except Exception:
            logging.exception(f"BENCHMARK RUN {run_number} / {n_instances} FAILED WITH AN EXCEPTION: {instance_file}")
            # Continue with the next instance.
            continue

        finally:
            # Flush and close the file handler before configuring the
            # logger for the next benchmark instance.
            logging.shutdown()

    print("\n" + "#" * 80)
    print("SOS1 BENCHMARK STUDY FINISHED")
    print("#" * 80)