from pathlib import Path
import json
from typing import Any, Dict, List
from dataclasses import asdict

from instance_generator import generate_instance, InstanceParameters

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
        instance_name: str,
        size_class: str,
        structural_regime: str,
        seed: int,
        instance_parameters: InstanceParameters,
        generator_version: str = "V3.0"
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
        params=instance_parameters,
        instance_name=instance_name,
        instance_size_class=size_class,
        instance_regime=structural_regime,
        seed=seed,
    )

    # Transport distance analysis
    summarize_distance_matrix("TD_gs (generation to transfer)", instance_data.TD_gs)
    summarize_distance_matrix("TD_si (transfer to incineration)", instance_data.TD_si)
    summarize_distance_matrix("TD_sl (transfer to landfill)", instance_data.TD_sl)
    summarize_distance_matrix("TD_sc (transfer to cement)", instance_data.TD_sc)

    metadata = {
        "Generator Version": generator_version,
        "Instance Name": instance_name,
        "Size Class": size_class,
        "Structural Regime": structural_regime,
        "Seed": seed,
        "Parameters": asdict(instance_parameters),

        "Dimensions": {
            "Generation spots": instance_data.G_max,
            "Transfer stations": instance_parameters.S_total,
            "Incineration plants": instance_parameters.I_total,
            "Landfill sites": instance_parameters.L_total,
            "Cement plants": instance_parameters.C_total,

            "Waste types": instance_data.W_max,
            "Capacity classes Co-Processing": instance_data.K_max,
            "Coal types": instance_data.F_max,
            "Subsidy levels": instance_data.H_max,
        },
        
        "Network Settings": {
            "City size x-axis (km)": instance_parameters.city_size_x,
            "City size y-axis (km)": instance_parameters.city_size_y,
            "City area (km²)": instance_parameters.city_size_x * instance_parameters.city_size_y,
            "Grid cell size (km)": instance_parameters.grid_cell_size,
            "Waste generation density (t/km²/year)": instance_parameters.waste_gen_density,

            "Incinerator radius - min (km)": instance_parameters.incinerator_radius_min,
            "Incinerator radius - max (km)": instance_parameters.incinerator_radius_max,
            "Incinerator radius - center (km)": instance_parameters.incinerator_radius_center,
            "Incinerator colocation probability": instance_parameters.incinerator_colocation_probability,
            "Landfill radius - min (km)": instance_parameters.landfill_radius_min,
            "Landfill radius - max (km)": instance_parameters.landfill_radius_max,
            "Landfill radius - center (km)": instance_parameters.landfill_radius_center,
            "Cement radius - min (km)": instance_parameters.cement_radius_min,
            "Cement radius - max (km)": instance_parameters.cement_radius_max,
            "Cement radius - center (km)": instance_parameters.cement_radius_center,
            "Clipping distances": instance_parameters.clipping_distances,
        },

        "Budget Availability": {
            "Municipality - Fraction of maximum Subsidy": instance_parameters.budget_mun_availability,
            "Cement - Fraction of maximum Co-Processing investment": instance_parameters.budget_cem_availability,
            "Subsidy Levels": instance_data.phi_wh,
        },

        # "Maximum subsidy levels (CNY/t) - [high moisture, medium moisture]": instance_parameters.phi_max_w,
        
        "Cost Parameters": {
            "Truck cost (CNY/t/km)": instance_parameters.c_truck,
            "Landfill cost (CNY/t)": instance_parameters.c_land,
            "Incineration cost (CNY/t)": instance_parameters.c_inc,
            "Coal price (CNY/t)": instance_parameters.price_coal_f,
            "Pre-processing cost (CNY/t) - [high moisture, medium moisture]": instance_parameters.c_preproc_w,
            "Penalty cost for denied allocated waste quota (CNY/t)": instance_parameters.c_penalty,
            "Investment cost for Co-Processing (CNY/capacity class)": instance_data.c_invest_k,
            "Investment-equivalent Co-Processing (CNY/year/capacity class)": instance_data.fixcost_invest_k,
        },

        "Emission Parameters": {
            "Truck emission (tCO2/t/km)": instance_data.epsilon_truck,
            "Landfill emission (tCO2/t)": instance_data.epsilon_land,
            "Incineration emission (tCO2/t)": instance_data.epsilon_inc,
            "Co-Processing emission - waste (tCO2/t)": instance_data.epsilon_kiln_w,
            "Co-Processing emission - coal (tCO2/t)": instance_data.epsilon_kiln_f,
        },

        "Disposal Limits": {
            "Landfill quota": instance_parameters.kappa_land,
            "Co-Processing quota": instance_parameters.kappa_coproc,
        },

        "Big-M unrestricted dual variables in KKT cuts": instance_parameters.bigM_duals,

        "Random Parameters": {
            "Waste Split range - high moisture (uniform, float)": instance_parameters.waste_split,  # range for high moisture split per node
            "Kiln energy consumption": instance_data.alpha_c,                                       # Triangular(7000, 18000, 15000) GJ per day * 365
        },

        "Calorific Values": {
            "Waste (GJ/t) - [high moisture, medium moisture]": instance_data.beta_w,
            "Coal (GJ/t) - coal type 1": instance_data.beta_f,
        },

        "Capacities": {
            "Waste generation (t/year/node)": instance_data.Q_gw,
            "Total waste generation (t/year)": instance_data.Q_gen_total,
            "Transfer station capacity (t/year)": instance_data.Q_s,
            "Landfill capacity (t/year)": instance_data.Q_l,
            "Incineration capacity (t/year)": instance_data.Q_i,
            "Co-Processing capacity (t/year/capacity class)": instance_data.Q_k,
        }
    }

    instance_data.instance_name = instance_name
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
    
    instance_parameters = InstanceParameters(
        S_total=8,
        I_total=6,
        L_total=2,
        C_total=4,

        city_size_x=40.0,
        city_size_y=30.0,
        grid_cell_size=10.0,
        waste_gen_density=2500,

        incinerator_radius_min=10.0,
        incinerator_radius_max=60.0,
        incinerator_radius_center=35.0,
        incinerator_colocation_probability=0.025,

        landfill_radius_min=40.0,
        landfill_radius_max=120.0,
        landfill_radius_center=65.0,

        cement_radius_min=80.0,
        cement_radius_max=250.0,
        cement_radius_center=150.0,

        clipping_distances=False,

        budget_mun_availability=0.80,
        budget_cem_availability=0.65,

        c_truck=0.45,
        c_land=180,
        c_inc=200,

        kappa_land=0.35,
        kappa_coproc=0.40,

        phi_max_w=[220.0, 175.0],

        price_coal_f=[700.0],

        c_preproc_w=[150.0, 125.0],
        c_penalty=100.0,
        bigM_duals=1e4,
    )

    output_path = write_instance_to_json(
        output_path=Path(__file__).parent / "generated_instances" / "medium" / instance_name,
        instance_name=instance_name[:-5],  # Remove ".json" extension
        size_class="medium",
        structural_regime="baseline",
        seed=7,
        instance_parameters=instance_parameters,
    )
    print(f"Instance generated and saved to {output_path}")

    # instance = read_instanceData_from_json(output_path)
    # print(instance)
#endregion