from pathlib import Path
from typing import Dict, Tuple

from instance_generator import InstanceParameters, compute_grid_generation_count
from json_writer import write_instance_to_json


"""
Batch instance generator.

Defines the experimental design (seeds × sizes × structural regimes) and
creates one JSON instance per combination via write_instance_to_json().
"""
####################################################################################
############################## Experimental design #################################
####################################################################################

SEEDS = [7, 19, 37, 61, 89]

SIZE_SETTINGS: Dict[str, dict] = {
    "small": {
        "city_size": 20.0,
        "radius_scale": 2 / 3,
    },
    "medium": {
        "city_size": 30.0,
        "radius_scale": 1.0,
    },
    "large": {
        "city_size": 40.0,
        "radius_scale": 4 / 3,
    },
}

REGIMES = ["balanced", "incDominated", "cemDominated"]

STRUCTURES: Dict[Tuple[str, str], dict] = {
    ("small", "balanced"):      dict(S_total=4,  I_total=3,  L_total=2, C_total=3),
    ("small", "incDominated"):  dict(S_total=4,  I_total=4,  L_total=2, C_total=2),
    ("small", "cemDominated"):  dict(S_total=4,  I_total=3,  L_total=1, C_total=5),

    ("medium", "balanced"):     dict(S_total=8,  I_total=6,  L_total=3, C_total=6),
    ("medium", "incDominated"): dict(S_total=8,  I_total=8,  L_total=3, C_total=4),
    ("medium", "cemDominated"): dict(S_total=8,  I_total=5,  L_total=2, C_total=8),

    ("large", "balanced"):      dict(S_total=12, I_total=9,  L_total=5, C_total=9),
    ("large", "incDominated"):  dict(S_total=12, I_total=12, L_total=5, C_total=6),
    ("large", "cemDominated"):  dict(S_total=12, I_total=8,  L_total=4, C_total=12),
}

# Canonical MEDIUM spatial structure (min, center, max) in km
BASE_RADII = {
    "inc":  (10.0, 35.0, 60.0),
    "land": (40.0, 65.0, 120.0),
    "cem":  (80.0, 150.0, 250.0),
}


####################################################################################
############################## Parameter builder ###################################
####################################################################################

def build_parameters(size: str, regime: str) -> InstanceParameters:
    """Construct InstanceParameters for a given (size, regime) combination."""
    size_cfg = SIZE_SETTINGS[size]
    structure = STRUCTURES[(size, regime)]
    f = size_cfg["radius_scale"]

    return InstanceParameters(
        # --- Network structure ---
        **structure,

        # --- Spatial layout ---
        city_size_x=size_cfg["city_size"],
        city_size_y=size_cfg["city_size"],
        grid_cell_size=10.0,
        waste_gen_density=3000,

        # --- Facility radii (scaled) ---
        incinerator_radius_min=BASE_RADII["inc"][0] * f,
        incinerator_radius_center=BASE_RADII["inc"][1] * f,
        incinerator_radius_max=BASE_RADII["inc"][2] * f,
        incinerator_colocation_probability=0.05,

        landfill_radius_min=BASE_RADII["land"][0] * f,
        landfill_radius_center=BASE_RADII["land"][1] * f,
        landfill_radius_max=BASE_RADII["land"][2] * f,

        cement_radius_min=BASE_RADII["cem"][0] * f,
        cement_radius_center=BASE_RADII["cem"][1] * f,
        cement_radius_max=BASE_RADII["cem"][2] * f,

        clipping_distances=False,

        # --- Budgets ---
        budget_mun_availability=0.80,
        budget_cem_availability=0.65,

        # --- Costs ---
        c_truck=0.45,
        c_land=180.0,
        c_inc=200.0,
        price_coal_f=[700.0],
        c_preproc_w=[150.0, 125.0],
        c_penalty=100.0,
        c_invest_k=[90_000_000.0, 120_000_000.0, 300_000_000.0],

        # Emission
        epsilon_truck=0.0002,
        epsilon_land=[0.85, 0.60],
        epsilon_inc=[0.20, 0.40],
        epsilon_kiln_w=[0.22, 0.68],
        epsilon_kiln_f=[2.25],

        # --- Policy / quotas ---
        kappa_land=0.35,
        kappa_coproc=0.40,
        phi_max_w=[220.0, 175.0],
        
        # --- Waste composition ---
        local_high_moisture_bounds_node=[0.4, 0.7],
        high_moisture_share_total_network=0.60,

        # --- Energy ---
        beta_f=[23.0],
        beta_w_nominal=[12.0, 16.0],
        moisture_reduction_factor_w=[0.60, 0.30],
        alpha_c_daily_min=7000.0,
        alpha_c_daily_max=18000.0,
        alpha_c_daily_mode=15000.0,

        # --- Capacity classes ---
        Q_s_classes=[182_500, 365_000, 500_000, 750_000, 1_000_000, 1_500_000],
        Q_i_classes=[365_000, 550_000, 750_000, 1_100_000, 1_850_000],
        Q_l_classes=[1_000_000, 2_000_000, 3_000_000],
        Q_k_daily=[350, 500, 1000],

        # --- Algorithmic (excluded from JSON metadata automatically) ---
        bigM_duals_unrestricted=1e4,
    )


####################################################################################
############################## Batch generation loop ###############################
####################################################################################

def generate_all_instances(
    output_root: Path | None = None,
    seeds: list[int] | None = None,
    sizes: list[str] | None = None,
    regimes: list[str] | None = None,
) -> list[Path]:
    """
    Generate all instances defined by the experimental design.

    Parameters
    ----------
    output_root : base directory for generated instances (default: ./generated_instances)
    seeds       : override SEEDS
    sizes       : subset of SIZE_SETTINGS keys (default: all)
    regimes     : subset of regimes (default: all three)

    Returns
    -------
    List of paths to the written JSON files.
    """
    if output_root is None:
        output_root = Path(__file__).parent / "Instances_CaseStudy_V1"
    if seeds is None:
        seeds = SEEDS
    if sizes is None:
        sizes = list(SIZE_SETTINGS.keys())
    if regimes is None:
        regimes = REGIMES

    written: list[Path] = []

    for size in sizes:
        for regime in regimes:
            if (size, regime) not in STRUCTURES:
                raise KeyError(f"No structure defined for ({size}, {regime})")
                # print(f"  [SKIP] No structure defined for ({size}, {regime})")
                # continue

            params = build_parameters(size, regime)

            for seed in seeds:
                size_letter = size[0].upper()
                instance_name = f"instance_{size_letter}_{regime}_seed_{seed:03d}.json"

                output_path=output_root / size / regime / instance_name

                if output_path.exists():
                    raise FileExistsError(f"Benchmark instance file already exists: {output_path}")

                written_path = write_instance_to_json(
                    output_path=output_path,
                    instance_name=instance_name[:-5],  # strip ".json"
                    size_class=size,
                    structural_regime=regime,
                    seed=seed,
                    instance_parameters=params,
                )
                written.append(written_path)
                print(f"  [OK] {written_path.relative_to(output_root)}")

    return written


####################################################################################
################################# Main entry #######################################
####################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("Batch instance generation")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Sizes:   {list(SIZE_SETTINGS.keys())}")
    print(f"  Regimes: {', '.join(REGIMES)}")
    print(f"  Total:   {len(SEEDS)} × {len(SIZE_SETTINGS)} × {len(REGIMES)} = {len(SEEDS) * len(SIZE_SETTINGS) * len(REGIMES)} instances")
    print("=" * 70)

    paths = generate_all_instances()

    expected_count = (
        len(SEEDS)
        * len(SIZE_SETTINGS)
        * len(REGIMES)
    )

    if len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} instances, "
            f"but generated {len(paths)}."
        )

    print(f"\nDone. {len(paths)} instances written.")