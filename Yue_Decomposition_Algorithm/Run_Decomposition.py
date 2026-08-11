import sys
from pathlib import Path
import logging
from datetime import datetime

# Ensure project root is importable (so "Instances" resolves)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Instances.json_reader import read_instance_data_from_json, read_instance_metadata_from_json
from Yue_Decomposition_Algorithm.Decomposition_Algorithm import run_yue_decomposition
from Yue_Decomposition_Algorithm.Normalization import determine_normalization_bounds


def setup_logger(instance_name: str, instance_size_class: str, instance_regime: str) -> Path:
    """Setup logging to file and console"""
    # Create solutions folder if it doesn't exist
    log_dir = Path(__file__).parent.parent / "Solutions" / instance_size_class / instance_regime
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with date and time
    now = datetime.now()
    log_filename = f"SOL_{instance_name}_{now.strftime('%Y%m%d_%H%M')}.log"
    log_path = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_path

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

if __name__ == "__main__":
    instance_path = Path(__file__).parent.parent / "Instances" / "generated_instances" / "medium"
    instance = read_instance_data_from_json(instance_path / "instance_m_base_002.json")
    instance_metadata = read_instance_metadata_from_json(instance_path / "instance_m_base_002.json")
    # instance_name = instance.instance_name
    # instance_name = instance_metadata['Instance Name']

    log_path = setup_logger(instance.instance_name, instance.instance_size_class, instance.instance_regime)
    logging.info(f"Yue-KKT Decomposition Algorithm started. Logs will be saved to {log_path}")
    log_instance_metadata(instance_metadata)

    normalization_bounds = determine_normalization_bounds(instance)
    
    solver_time_limit = 500     # Time limit for solving MP every 5th iteration (in seconds)
    mip_gap = 1e-6              # MIP gap for the master problem
    Xi = 1e-4                   # Convergence threshold for leader objective improvement
    max_iterations = 5         # Maximum number of iterations to prevent infinite loops
    total_runtime = 3630        # Total runtime limit for the entire decomposition algorithm (in seconds)

    run_yue_decomposition(
        Verbose=True, 
        solver_time_limit=solver_time_limit, 
        mip_gap=mip_gap, 
        Xi=Xi, 
        max_iterations=max_iterations, 
        instance=instance,
        weight_env=1.0,
        weight_mon=1.0,
        total_time_limit=total_runtime,
        objective_scale=100.0,
        sos1_cuts=True,
        primal_dual_strenghtening=True,
        bound_cutoff=True,
        cutoff_bound_tolerance=1e-5,
    )