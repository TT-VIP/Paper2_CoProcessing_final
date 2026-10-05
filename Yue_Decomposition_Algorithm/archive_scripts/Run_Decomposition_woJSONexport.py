import sys
from pathlib import Path
import logging
from datetime import datetime
import math

# Ensure project root is importable (so "Instances" resolves)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Instances.json_reader import read_instance_data_from_json, read_instance_metadata_from_json
from Yue_Decomposition_Algorithm.Decomposition_Algorithm_woIterationRecord import run_yue_decomposition
from Yue_Decomposition_Algorithm.Normalization import determine_normalization_bounds
# from Yue_Decomposition_Algorithm.Lexico_Normalization import determine_normalization_bounds

###############################################################################################
################################# Helper Functions ############################################
###############################################################################################
#region Helper functions
def setup_logger(
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
    base_dir = Path(__file__).parent.parent / "Solutions" / instance_size_class / instance_regime
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with date and time
    now = datetime.now()
    log_filename = f"SOL_{instance_name}_{method_tag}_{now.strftime('%Y%m%d_%H%M')}"

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
            logging.FileHandler(log_path, encoding='utf-8'),
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
    lb_stall_trigger: int,
    mip_gap: float,
    Xi: float,
    max_iterations: int,
    total_runtime: float,
    weight_env: float,
    weight_mon: float,
    objective_scale: float,
    sos1_cuts: bool,
    primal_dual_strenghtening: bool,
    bigM_duals: float,
    bound_cutoff: bool,
    cutoff_bound_tolerance: float,
) -> None:
    """Log all algorithmic settings defining the solver run."""

    logging.info("=" * 70)
    logging.info("SOLVER RUN CONFIGURATION")
    logging.info("=" * 70)

    logging.info(f"Xi: {Xi:g}")
    logging.info(f"Master MIP-gap (Gurobi): {mip_gap:g}")
    logging.info(f"Maximum decomposition iterations: {max_iterations}")
    logging.info(f"Total runtime limit: {total_runtime:g} s")
    logging.info(f"MP normal time limit: {mp_normal_time_limit:g} s")
    logging.info(f"MP polishing time limit: {mp_polish_time_limit:g} s")
    logging.info(f"LB stall trigger: {lb_stall_trigger}")
    logging.info(f"SP1 max time: {sp1_max_time:g} s")
    logging.info(f"SP2 max time: {sp2_max_time:g} s")

    logging.info(
        f"Objective weights: "
        f"environment={weight_env:g}, monetary={weight_mon:g}"
    )
    logging.info(f"Objective scale: {objective_scale:g}")

    if sos1_cuts:
        logging.info("Complementarity formulation: SOS1")
        logging.info(
            "Primal-dual strengthening: "
            f"{'enabled' if primal_dual_strenghtening else 'disabled'}"
        )
        logging.info("Generic dual Big-M: N/A")
    else:
        logging.info("Complementarity formulation: Big-M")
        logging.info(f"Generic dual Big-M: {bigM_duals:.6g}")
        logging.info("Primal-dual strengthening: N/A")

    logging.info(
        f"Bound cutoff: {'enabled' if bound_cutoff else 'disabled'}"
    )

    if bound_cutoff:
        logging.info(
            f"Bound cutoff tolerance: {cutoff_bound_tolerance:g}"
        )
    else:
        logging.info("Bound cutoff tolerance: N/A")

    logging.info("=" * 70)

def format_bigM(value: float) -> str:
    """Format positive powers of ten cleanly for filenames."""
    if value <= 0:
        raise ValueError("Big-M value must be positive.")

    exponent = int(round(math.log10(value)))

    if math.isclose(value, 10 ** exponent):
        return f"1e{exponent}"

    return f"{value:g}"

def get_method_tag(
    sos1_cuts: bool,
    primal_dual_strenghtening: bool,
    bigM_duals: float,
) -> str:
    if sos1_cuts:
        return (
            "SOS1+Strengthened"
            if primal_dual_strenghtening
            else "SOS1"
        )

    return f"BigM{format_bigM(bigM_duals)}"

###############################################################################################
#################################### Run Algorithm ############################################
###############################################################################################
# Run algorithm for one specific instance
if __name__ == "__main__":
    # ============================================================
    # Load instance
    # ============================================================
    instance_path = Path(__file__).parent.parent / "Instances" / "generated_instances" / "medium" / "incDominated"
    instance_file = instance_path / "instance_M_incDominated_seed_007.json"
    instance = read_instance_data_from_json(instance_file)
    instance_metadata = read_instance_metadata_from_json(instance_file)

    # ============================================================
    # Solver / decomposition configuration
    # ============================================================
    mp_normal_time_limit = 180          # Time limit for solving MP for exploration (in seconds)
    mp_polish_time_limit = 600          # Time limit for solving MP for polishing (in seconds)
    sp1_max_time = 60                   # Time limit for solving SP1 (in seconds)
    sp2_max_time = 60                   # Time limit for solving SP2 (in seconds)

    lb_stall_trigger = 2            # Number of consecutive iterations with no meaningful LB improvement to trigger polishing MP strategy
    mip_gap = 1e-4                  # MIP gap for the master problem
    Xi = 1e-4                       # Convergence threshold for leader objective improvement
    max_iterations = 7              # Maximum number of iterations to prevent infinite loops
    total_runtime = 3630            # Total runtime limit for the entire decomposition algorithm (in seconds)

    weight_env = 1.0                    # Weighting factor for the environmental emission objective in the leader's objective function (for weighted-sum approach)
    weight_mon = 1.0                    # Weighting factor for the monetary cost objective in the leader's objective function (for weighted-sum approach)
    objective_scale = 100.0             # Scale factor for the leader objective to avoid numerical solver issues in the master problem

    sos1_cuts = False                   # Whether to use SOS1 cuts in the master problem
    primal_dual_strenghtening = True    # Whether to use primal-dual strengthening in the master problem

    bigM_duals = 1e5                    # Big-M value for unrestricted dual variables in KKT cuts

    bound_cutoff = True                 # Whether to use bound cutoff in the consecutive iterations of the master problem
    cutoff_bound_tolerance = 1e-5       # Tolerance for bound cutoff

    # ============================================================
    # Logging
    # ============================================================
    method_tag = get_method_tag(
        sos1_cuts=sos1_cuts,
        primal_dual_strenghtening=primal_dual_strenghtening,
        bigM_duals=bigM_duals,
    )
    
    log_path, run_dir = setup_logger(
        instance_name = instance.instance_name,
        instance_size_class = instance.instance_size_class,
        instance_regime = instance.instance_regime,
        method_tag = method_tag,
    )

    logging.info(f"Yue-KKT Decomposition Algorithm started. Logs will be saved to {log_path}")
    logging.info(f"Infeasible SP2 IIS-files will be saved to {run_dir / 'SP2_IIS'}")

    log_run_metadata(
        mp_normal_time_limit=mp_normal_time_limit,
        mp_polish_time_limit=mp_polish_time_limit,
        sp1_max_time=sp1_max_time,
        sp2_max_time=sp2_max_time,
        lb_stall_trigger=lb_stall_trigger,
        mip_gap=mip_gap,
        Xi=Xi,
        max_iterations=max_iterations,
        total_runtime=total_runtime,
        weight_env=weight_env,
        weight_mon=weight_mon,
        objective_scale=objective_scale,
        sos1_cuts=sos1_cuts,
        primal_dual_strenghtening=primal_dual_strenghtening,
        bigM_duals=bigM_duals,
        bound_cutoff=bound_cutoff,
        cutoff_bound_tolerance=cutoff_bound_tolerance,
    )

    logging.info("")
    log_instance_metadata(instance_metadata)

    # ============================================================
    # Normalization
    # ============================================================
    normalization_bounds = determine_normalization_bounds(instance)

    # ============================================================
    # Decomposition
    # ============================================================
    run_yue_decomposition(
        Verbose=True, 
        mp_normal_time_limit=mp_normal_time_limit, 
        mp_polish_time_limit=mp_polish_time_limit, 
        sp1_max_time=sp1_max_time, 
        sp2_max_time=sp2_max_time, 
        lb_stall_trigger=lb_stall_trigger,
        mip_gap=mip_gap, 
        Xi=Xi, 
        max_iterations=max_iterations, 
        instance=instance,
        weight_env=weight_env,
        weight_mon=weight_mon,
        total_time_limit=total_runtime,
        objective_scale=objective_scale,
        bigM_duals_unrestricted=bigM_duals,
        sos1_cuts=sos1_cuts,
        primal_dual_strenghtening=primal_dual_strenghtening,
        bound_cutoff=bound_cutoff,
        cutoff_bound_tolerance=cutoff_bound_tolerance,
        solution_dir=run_dir,
    )

# Run algorithm for all instances in a folder
# if __name__ == "__main__":
#     instance_path = Path(__file__).parent.parent / "Instances" / "generated_instances" / "medium" / "baseline"

#     solver_time_limit = 500     # Time limit for solving MP every 5th iteration (in seconds)
#     mip_gap = 1e-6              # MIP gap for the master problem
#     Xi = 1e-4                   # Convergence threshold for leader objective improvement
#     max_iterations = 5          # Maximum number of iterations to prevent infinite loops
#     total_runtime = 3630

#     for instance_file in sorted(instance_path.glob("*.json")):
#         instance = read_instance_data_from_json(instance_file)
#         instance_metadata = read_instance_metadata_from_json(instance_file)

#         log_path = setup_logger(instance.instance_name, instance.instance_size_class, instance.instance_regime)
#         logging.info(f"Yue-KKT Decomposition Algorithm started for instance {instance.instance_name}. Logs will be saved to {log_path}")
#         log_instance_metadata(instance_metadata)

#         normalization_bounds = determine_normalization_bounds(instance)

#         run_yue_decomposition(
#             Verbose=True, 
#             solver_time_limit=solver_time_limit, 
#             mip_gap=mip_gap, 
#             Xi=Xi, 
#             max_iterations=max_iterations, 
#             instance=instance,
#             weight_env=1.0,
#             weight_mon=1.0,
#             total_time_limit=total_runtime,
#             objective_scale=100.0,
#             sos1_cuts=True,
#             primal_dual_strenghtening=True,
#             bound_cutoff=True,
#             cutoff_bound_tolerance=1e-5,
#         )