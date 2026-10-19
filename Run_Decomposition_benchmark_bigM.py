import sys
from pathlib import Path
import logging
import math

# Ensure project root is importable (so "Instances" resolves)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Instances.json_reader import read_instance_data_from_json, read_instance_metadata_from_json
from Yue_Decomposition_Algorithm.Decomposition_Algorithm import run_yue_decomposition
from Yue_Decomposition_Algorithm.Normalization import determine_normalization_bounds
from Yue_Decomposition_Algorithm.Bilevel_solution_export import build_solution_dictionary, write_solution_json


# =============================================================================
# Calibration-study design
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

# Output separately to avoid overwriting the original 135-run calibration.
STUDY_OUTPUT_FOLDER = "BigM"
INSTANCE_FOLDER = "Instances_CaseStudy_V3"

DUAL_KEYS_TO_COMPARE = ("lam_F4", "lam_F5", "lam_F6", "pi_q_scw")

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
    
    # Create log filename with date and time
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
    sos1_cuts: bool,
    primal_dual_strenghtening: bool,
    bigM_duals: float | None,
    use_analytical_dual_bounds: bool,
    analytical_dual_bound_margin: float | None,
    effective_dual_bigM: dict | None,
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

    if sos1_cuts:
        logging.info("Complementarity formulation: SOS1")
        logging.info(f"Primal-dual strengthening: {primal_dual_strenghtening}")
    else:
        logging.info("Complementarity formulation: Big-M")
        if use_analytical_dual_bounds:
            logging.info("Analytical dual big-M applied.")
            logging.info(f"Analytical dual margin (above derived B): {analytical_dual_bound_margin:g}")
        else:
            logging.info(f"Generic dual Big-M: {bigM_duals:.6g}")
        logging.info("Effective dual Big-M values:")
        for key, value in sorted(effective_dual_bigM.items()):
            logging.info(f"  {key}: {value:.6g}")

    logging.info(f"Bound cutoff: {'enabled' if bound_cutoff else 'disabled'}")
    logging.info(
        f"Bound cutoff tolerance: {cutoff_bound_tolerance:g}"
        if bound_cutoff else "Bound cutoff tolerance: N/A"
    )
    logging.info("=" * 70)

def format_power_of_ten(value: float) -> str:
    """Format positive powers of ten cleanly for filenames, e.g. 1e-5, 1e4."""
    if value <= 0:
        raise ValueError("Big-M value must be positive.")

    exponent = int(round(math.log10(value)))

    if math.isclose(value, 10 ** exponent):
        return f"1e{exponent}"

    return f"{value:g}"

def get_method_tag(
    sos1_cuts: bool,
    primal_dual_strenghtening: bool,
    bigM_duals: float | None,
    cutoff_bound_tolerance: float,
    use_analytical_dual_bounds: bool,
    analytical_dual_bound_margin: float | None,
) -> str:
    if sos1_cuts:
        formulation_tag = "SOS1-Strength" if primal_dual_strenghtening else "SOS1"
    elif use_analytical_dual_bounds:
        if analytical_dual_bound_margin is None:
            raise ValueError("An analytical margin is required for analytical Big-M.")
        if math.isclose(analytical_dual_bound_margin, 1.0):
            formulation_tag = "BigM-analytic"
        else:
            formulation_tag = f"BigM-analytic{analytical_dual_bound_margin:g}"
    else:
        if bigM_duals is None:
            raise ValueError("bigM_duals is required for generic Big-M.")
        formulation_tag = f"BigM{format_power_of_ten(bigM_duals)}"

    return f"{formulation_tag}_cutTol{format_power_of_ten(cutoff_bound_tolerance)}"


def configure_dual_bounds(
    instance,
    *,
    use_analytical_dual_bounds: bool,
    bigM_duals: float | None,
    analytical_dual_bound_margin: float | None,
) -> dict[str, float]:
    """Apply the chosen margin to dual bounds.

    The generator stores B_j + 1. Margin m means B_j + m, hence add (m - 1).
    No change is made to lam_F3, pi_q_cf, or any primal Big-M bound.
    """
    missing = set(DUAL_KEYS_TO_COMPARE) - set(instance.M_dual)
    if missing:
        raise KeyError(f"Missing dual Big-M keys: {sorted(missing)}")

    if use_analytical_dual_bounds:
        if bigM_duals is not None:
            raise ValueError("Set bigM_duals=None in analytical mode.")
        if (analytical_dual_bound_margin is None
                or not math.isfinite(analytical_dual_bound_margin)
                or analytical_dual_bound_margin < 0):
            raise ValueError("An analytical margin >= 0 is required.")
        increment = analytical_dual_bound_margin - 1.0
        for key in DUAL_KEYS_TO_COMPARE:
            instance.M_dual[key] = float(instance.M_dual[key]) + increment
    else:
        if (bigM_duals is None or not math.isfinite(bigM_duals)
                or bigM_duals <= 0):
            raise ValueError("A positive finite generic Big-M is required.")
        for key in DUAL_KEYS_TO_COMPARE:
            instance.M_dual[key] = float(bigM_duals)

    if any(value < 0 or not math.isfinite(value) for value in instance.M_dual.values()):
        raise ValueError("Effective dual Big-M bounds must be finite and nonnegative.")
    return dict(instance.M_dual)

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
    # Parameters remain identical across calibration study.
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

        # IMPORTANT:
        # Set this to the intended calibration-study value.
        # 2 is suitable only for a smoke test.
        "max_iterations": 100,

        "sos1_cuts": False,
        "primal_dual_strenghtening": False,
        "bigM_duals": None,
        "use_analytical_dual_bounds": True,
        "analytical_dual_bound_margin": 1,
        "cutoff_bound_tolerance": 1e-3,

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

        print("\n" + "#" * 80)
        print(f"INSTANCE: {instance_file.name}")
        print("#" * 80)

        # ---------------------------------------------------------------------
        # Determine normalization bounds ONCE for this instance
        # ---------------------------------------------------------------------
        normalization_instance = read_instance_data_from_json(instance_file)

        print(f"Computing normalization bounds for {normalization_instance.instance_name} ...")
        determine_normalization_bounds(normalization_instance)

        normalization_bounds = extract_normalization_bounds(normalization_instance)
        instance_name = normalization_instance.instance_name
        print(
            "Normalization bounds:"
            f"\n  Emission min = {normalization_bounds['emission_min']}"
            f"\n  Emission max = {normalization_bounds['emission_max']}"
            f"\n  Cost min     = {normalization_bounds['cost_min']}"
            f"\n  Cost max     = {normalization_bounds['cost_max']}"
        )

        # Metadata itself is identical across the configurations.
        instance_metadata = read_instance_metadata_from_json(instance_file)

        # ---------------------------------------------------------------------
        # Run the algorithm configurations
        # ---------------------------------------------------------------------

        method_tag = get_method_tag(
            sos1_cuts=(FIXED_ALGORITHM_CONFIG["sos1_cuts"]),
            primal_dual_strenghtening=(FIXED_ALGORITHM_CONFIG["primal_dual_strenghtening"]),
            bigM_duals=(FIXED_ALGORITHM_CONFIG["bigM_duals"]),
            cutoff_bound_tolerance=(FIXED_ALGORITHM_CONFIG["cutoff_bound_tolerance"]),
            use_analytical_dual_bounds=(FIXED_ALGORITHM_CONFIG["use_analytical_dual_bounds"]),
            analytical_dual_bound_margin=(FIXED_ALGORITHM_CONFIG["analytical_dual_bound_margin"]),
        )

        print("\n" + "=" * 80)
        print(f"RUN {run_number:03d}/{n_instances}: {instance_name} | {method_tag}")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # Reload a pristine instance for every run.
        #
        # run_yue_decomposition() modifies some algorithm-dependent attributes
        # ---------------------------------------------------------------------
        instance = read_instance_data_from_json(instance_file)
        apply_normalization_bounds(instance, normalization_bounds)

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        log_path, run_dir = setup_logger(
            instance_name=instance.instance_name,
            instance_size_class=instance.instance_size_class,
            instance_regime=instance.instance_regime,
            method_tag=method_tag,
        )

        effective_dual_bigM = configure_dual_bounds(
            instance,
            use_analytical_dual_bounds=(FIXED_ALGORITHM_CONFIG["use_analytical_dual_bounds"]),
            bigM_duals=(FIXED_ALGORITHM_CONFIG["bigM_duals"]),
            analytical_dual_bound_margin=(FIXED_ALGORITHM_CONFIG["analytical_dual_bound_margin"]),
        )

        logging.info("=" * 80)
        logging.info("COMPUTATIONAL STUDY - BENCHMARK")
        logging.info("=" * 80)
        logging.info(f"Overall run: {run_number}/{n_instances}")
        logging.info(f"Configuration: {method_tag}")
        logging.info(f"Instance file: {instance_file}")
        logging.info("=" * 80)

        logging.info(f"Yue-KKT Decomposition Algorithm started. Log: {log_path}")
        logging.info(f"Infeasible SP2 IIS-files will be saved to {run_dir / 'SP2_IIS'}")

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
            sos1_cuts=(FIXED_ALGORITHM_CONFIG["sos1_cuts"]),
            primal_dual_strenghtening=(FIXED_ALGORITHM_CONFIG["primal_dual_strenghtening"]),
            bigM_duals=(FIXED_ALGORITHM_CONFIG["bigM_duals"]),
            use_analytical_dual_bounds=(FIXED_ALGORITHM_CONFIG["use_analytical_dual_bounds"]),
            analytical_dual_bound_margin=(FIXED_ALGORITHM_CONFIG["analytical_dual_bound_margin"]),
            effective_dual_bigM=effective_dual_bigM,
            bound_cutoff=FIXED_ALGORITHM_CONFIG["bound_cutoff"],
            cutoff_bound_tolerance=(FIXED_ALGORITHM_CONFIG["cutoff_bound_tolerance"]),
        )

        logging.info("")
        log_instance_metadata(instance_metadata)

        # ---------------------------------------------------------------------
        # Decomposition
        # ---------------------------------------------------------------------
        decomposition_config = {
            key: value
            for key, value in FIXED_ALGORITHM_CONFIG.items()
            if key not in {"bigM_duals","analytical_dual_bound_margin"}     # both arguments are not within run_yue_decomposition, thatswhy filter them
        }
        try:
            decomposition_solution = run_yue_decomposition(
                **decomposition_config,

                instance=instance,
                bigM_duals_unrestricted=None,

                solution_dir=run_dir,
            )

            algorithm_config = {
                "study": "Benchmark_BigM",
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
                "bigM_duals_unrestricted": FIXED_ALGORITHM_CONFIG["bigM_duals"],
                "use_analytical_dual_bounds": FIXED_ALGORITHM_CONFIG["use_analytical_dual_bounds"],
                "analytical_dual_bound_margin": FIXED_ALGORITHM_CONFIG["analytical_dual_bound_margin"],
                "effective_dual_bigM": dict(instance.M_dual),

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
            logging.exception("CALIBRATION RUN FAILED WITH AN EXCEPTION.")
            # Continue with the next configuration.
            continue

        finally:
            # Flush and close the file handler before configuring the
            # logger for the next calibration run.
            logging.shutdown()

    print("\n" + "#" * 80)
    print("CALIBRATION STUDY FINISHED")
    print("#" * 80)