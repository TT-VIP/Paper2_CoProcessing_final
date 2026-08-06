from pathlib import Path
import json
from typing import Any, Dict, List
from dataclasses import asdict

from Instances.archive_scripts.instance_generator import generate_instance

#region Helper Functions
# JSON serialization
def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, range):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(value) for value in obj]
    return obj

# Distance matrix report
def summarize_distance_matrix(name: str, matrix: List[List[int]]) -> None:
    """
    Print basic diagnostics for a distance matrix.
    """
    values = [value for row in matrix for value in row]
    values_sorted = sorted(values)

    n = len(values_sorted)

    def quantile(q: float) -> int:
        index = int(round(q * (n - 1)))
        return values_sorted[index]

    print(f"\n{name}")
    print(f"  min:    {min(values)} km")
    print(f"  q25:    {quantile(0.25)} km")
    print(f"  mean:   {sum(values) / n:.2f} km")
    print(f"  median: {quantile(0.50)} km")
    print(f"  q75:    {quantile(0.75)} km")
    print(f"  max:    {max(values)} km")
#endregion

#region JSON writer
def write_instance_to_json(
        output_path: Path,
        instance_id: str,
        size_class: str,
        structural_regime: str,
        seed: int,
        instance_parameters: Dict[str, Any],
        generator_version: str = "V2.0"
) -> Path:
    '''
    Generate an instance and write it to a JSON file with separated metadata and data:
    {
    "metadata": {
        "instance_id": "instance_001",
        "size_class": "small",
        "structural_regime": "balanced",
        "seed": 7,
        "distribution_settings": {...}
    },
    "data": {
        ... instance data fields ...
    }
    }
    '''
    instance_data = generate_instance(
        seed=seed,
        **instance_parameters
    )
    # instance_data = generate_instance(
    #     S_total=instance_parameters["S_total"],
    #     I_total=instance_parameters["I_total"],
    #     L_total=instance_parameters["L_total"],
    #     C_total=instance_parameters["C_total"],

    #     city_size=instance_parameters["city_size"],
    #     grid_cell_size=instance_parameters["grid_cell_size"],
    #     waste_gen_density=instance_parameters["waste_gen_density"],

    #     incinerator_radius_min=instance_parameters["incinerator_radius_min"],
    #     incinerator_radius_max=instance_parameters["incinerator_radius_max"],
    #     incinerator_radius_center=instance_parameters["incinerator_radius_center"],
    #     incinerator_colocation_probability=instance_parameters["incinerator_colocation_probability"],
    #     landfill_radius_min=instance_parameters["landfill_radius_min"],
    #     landfill_radius_max=instance_parameters["landfill_radius_max"],
    #     landfill_radius_center=instance_parameters["landfill_radius_center"],
    #     cement_radius_min=instance_parameters["cement_radius_min"],
    #     cement_radius_max=instance_parameters["cement_radius_max"],
    #     cement_radius_center=instance_parameters["cement_radius_center"],

    #     clipping_distances=instance_parameters["clipping_distances"],
    #     seed=seed,
    # )

    # Transport distance analysis
    summarize_distance_matrix("TD_gs (generation to transfer)", instance_data.TD_gs)
    summarize_distance_matrix("TD_si (transfer to incineration)", instance_data.TD_si)
    summarize_distance_matrix("TD_sl (transfer to landfill)", instance_data.TD_sl)
    summarize_distance_matrix("TD_sc (transfer to cement)", instance_data.TD_sc)

    metadata = {
        "instance_id": instance_id,
        "size_class": size_class,
        "structural_regime": structural_regime,
        "seed": seed,

        "dimensions": {
            "G_total": instance_data.G_max,
            "S_total": instance_parameters["S_total"],
            "I_total": instance_parameters["I_total"],
            "L_total": instance_parameters["L_total"],
            "C_total": instance_parameters["C_total"],
        },
        
        "network_settings": {
            "city_size_x": instance_parameters["city_size_x"],
            "city_size_y": instance_parameters["city_size_y"],
            "city_area": instance_parameters["city_size_x"] * instance_parameters["city_size_y"],
            "grid_cell_size": instance_parameters["grid_cell_size"],
            "waste_gen_density": instance_parameters["waste_gen_density"],

            "incinerator_radius_min": instance_parameters["incinerator_radius_min"],
            "incinerator_radius_max": instance_parameters["incinerator_radius_max"],
            "incinerator_radius_center": instance_parameters["incinerator_radius_center"],
            "incinerator_colocation_probability": instance_parameters["incinerator_colocation_probability"],
            "landfill_radius_min": instance_parameters["landfill_radius_min"],
            "landfill_radius_max": instance_parameters["landfill_radius_max"],
            "landfill_radius_center": instance_parameters["landfill_radius_center"],
            "cement_radius_min": instance_parameters["cement_radius_min"],
            "cement_radius_max": instance_parameters["cement_radius_max"],
            "cement_radius_center": instance_parameters["cement_radius_center"],
            "clipping_distances": instance_parameters["clipping_distances"],
        },

        "waste_split (uniform, float)": [0.4, 0.7],  # range for high moisture split
        "alpha_c (triangular, int)": [7000, 18000, 15000],
        "budget_availability_municipality": instance_parameters["budget_mun_availability"],   # 80% of the maximum total potential waste flow to kilns can be subsidized supposing maximum subsidy levels
        "budget_availability_cement": instance_parameters["budget_cem_availability"],         # Only 65% of the maximum total potential investment cost for co-processing is available
        "generator_version": generator_version
    }

    instance_data.instance_name = instance_id
    instance_data.instance_size_class = size_class
    instance_data.instance_regime = structural_regime

    payload = {
        "metadata": metadata,
        "data": _make_json_serializable(asdict(instance_data))
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    
    return output_path
#endregion

#region Create JSON instance
# call the script to generate an instance and save to JSON within the python environment (can be adapted to command-line arguments if needed)
if __name__ == "__main__":
    instance_name = "instance_m_base_002.json"
    
    instance_parameters = {
        "S_total": 8,
        "I_total": 6,
        "L_total": 2,
        "C_total": 4,

        "city_size_x": 40.0,
        "city_size_y": 30.0,
        "grid_cell_size": 10.0,
        "waste_gen_density": 2500,

        "incinerator_radius_min": 10.0,
        "incinerator_radius_max": 60.0,
        "incinerator_radius_center": 35.0,
        "incinerator_colocation_probability": 0.025,

        "landfill_radius_min": 40.0,
        "landfill_radius_max": 120.0,
        "landfill_radius_center": 65.0,
        
        "cement_radius_min": 80.0,
        "cement_radius_max": 250.0,
        "cement_radius_center": 150.0,

        "clipping_distances": False,
    }

    output_path = write_instance_to_json(
        output_path=Path(__file__).parent / "generated_instances" / "medium" / instance_name,
        instance_id=instance_name[:-5],  # Remove ".json" extension
        size_class="medium",
        structural_regime="baseline",
        seed=42,
        instance_parameters=instance_parameters,
    )
    print(f"Instance generated and saved to {output_path}")

    # instance = read_instanceData_from_json(output_path)
    # print(instance)
#endregion