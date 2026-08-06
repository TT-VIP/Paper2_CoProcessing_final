from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Dict, Tuple, Any, Literal


#####################################################################################
################################ Data class #########################################
#####################################################################################
#region Instance data class
@dataclass(frozen=False)
class InstanceData:
    # Instance metadata
    instance_name: str
    instance_size_class: str
    instance_regime: str
    
    # Sets (sizes)
    G_max: int      # Number of Generation spots
    S_max: int      # Number of Transfer stations
    W_max: int      # Number of Waste types
    I_max: int      # Number of Incinerators
    L_max: int      # Number of Landfills
    C_max: int      # Number of Cement facilities
    K_max: int      # Number of available Pre- and Co-processing capacities
    F_max: int      # Number of Coal types as conservative fuel
    H_max: int      # Number of Subsidy levels

    # Index sets
    G: range        # Set of Generation spots
    S: range        # Set of Transfer stations
    W: range        # Set of Waste types
    I: range        # Set of Incinerators
    L: range        # Set of Landfills
    C: range        # Set of Cement facilities
    K: range        # Set of available Pre- and Co-processing capacities
    F: range        # Set of Coal types as conservative fuel
    H: range        # Set of Subsidy levels 

    # Names (optional)
    cement_names: List[str]

    # Coordinates
    G_coords: CoordinateDict  # Coordinates of Generation spots
    S_coords: CoordinateDict  # Coordinates of Transfer stations
    I_coords: CoordinateDict  # Coordinates of Incinerators
    L_coords: CoordinateDict  # Coordinates of Landfills
    C_coords: CoordinateDict  # Coordinates of Cement facilities

    # Distances (km)
    TD_gs: List[List[int]]      # TD[g][s] Transportation distance from Generation to Transfer
    TD_sl: List[List[int]]      # TD[s][l] Transportation distance from Transfer to Landfill
    TD_si: List[List[int]]      # TD[s][i] Transportation distance from Transfer to Incinerator
    TD_si_avg: float            # Average transportation distance to incinerator
    TD_sc: List[List[int]]      # TD[s][c] Transportation distance from Transfer to Cement facility

    # Leader parameters
    epsilon_truck: float            # Emission factor for trucks (ton CO2e per ton-km)
    epsilon_land: List[float]       # Emission factors for landfills (ton CO2e per ton)
    epsilon_inc: List[float]        # Emission factors for incinerators (ton CO2e per ton)
    epsilon_kiln_w: List[float]     # Emission factors for kiln waste (ton CO2e per ton)
    epsilon_kiln_f: List[float]     # Emission factors for kiln fuel (ton CO2e per ton)  
    c_truck: float          # Transportation cost (CNY/ton-km)
    c_land: float           # Landfill cost (CNY/ton)
    c_inc: float            # Incineration cost (CNY/ton)

    # Waste quantities (t/year)
    Q_gw: List[List[int]]       # Q[g][w] Waste quantity at Generation spots
    Q_gen_total: int            # Total generated waste (t/year)
    Q_s: List[int]              # Q[s] Capacity at Transfer stations
    Q_l: List[int]              # Q[l] Capacity at Landfills
    Q_i: List[int]              # Q[i] Capacity at Incinerators

    # Co-processing capacity options (t/year)
    Q_k: List[int]              # Q[k] Available Pre- & Co-processing capacities or investment at Cement facilities
    Q_k_max: int                # Max cement kiln capacity for co-procesing (tons)

    # Objective weights & policy
    kappa_land: float               # Maximum allowed landfill quota/capacity
    kappa_coproc: float             # Maximum co-processing quota/capacity
    budget_municipality: float      # Budget for the municipality
    phi_max: List[float]            # Maximum subsidy levels for waste types (CNY/ton)
    phi_wh: List[List[float]]       # phi[w][h] Subsidy levels for waste types
    
    # Follower parameters
    price_f: List[float]        # p[f] Price of coal types (CNY/t)
    beta_f: List[float]         # beta[f] Calorific value of coal types (GJ/t)
    alpha_c: List[float]        # alpha[c] Energy requirement of cement kiln (GJ/period) (or consistent with your model)
    beta_w: List[float]         # beta[w] Calorific value of waste types (GJ/t)
    # eta_w: List[float]          # eta[w] Weight reductio after pre-processing for waste type w (0 < eta_w <= 1, where 1 means no reduction)

    c_invest_k: List[float]         # c_invest[k] Investment cost for Pre- & Co-processing facility per capacity (CNY)
    c_preproc_w: List[float]        # c_preproc[w] Pre-processing cost per waste type (CNY/t)
    c_penalty: float                # Penalty cost for denying allocated waste quota (CNY/t)
    budget_cem: float               # Budget of the cement producers (CNY)

    tau: float            # Symmetry breaking parameter for follower problem

    # Fix-cost invest-equivalent per capacity (CNY/year)
    fixcost_invest_k: List[float]

    # Big-M values for primal and dual variables in KKT cuts
    M_primal: Dict[str, float | Dict[int | float, Any]]   # Big-M values for primal variables in KKT cuts (indexed dictionary by s for r_sw)
    M_dual: Dict[str, float]

    U_w: List[int]          # Upper bound on waste flow of type w (can be tightened based on data)

    # multi-objective weights and bounds
    weight_env: float = 0.5               # Weight for environmental objective in leader problem
    weight_mon: float = 0.5               # Weight for monetary objective in leader problem
    Emission_min: float | None = None            # Minimum emissions (single objective for normalization)
    Emission_max: float | None = None            # Maximum emissions (single objective for normalization)
    Cost_min: float | None = None            # Minimum cost (single objective for normalization)
    Cost_max: float | None = None            # Maximum cost (single objective for normalization)
#endregion

####################################################################################
############################## Helper functions ####################################
####################################################################################
# region Helper functions

# Capital Recovery Factor for annualizing investment costs: 
# CRF(i,n) = (i*(1+i)^n)/((1+i)^n-1) 
# where i is the interest rate and n is the number of periods
# used to convert an upfront investment cost into an equivalent periodic cost for comparison with operational costs in the objective function
def crf(i: float, n: int) -> float:
    return (i * (1 + i) ** n) / ((1 + i) ** n - 1)

Point = Tuple[float, float]
CoordinateDict = Dict[int, Point]

#region Generation points
# Fixed demand grid based on city size and cell size parameters, because waste is generated by all households across the city 
# and thus the generation points are not really random, but rather determined by the urban structure; this also keeps the TD 
# matrices consistent across different runs and allows for more meaningful analysis of the results
def generate_waste_generation_points(
    city_size_x: float,
    city_size_y: float,
    cell_size: float,
    center: Point = (0.0, 0.0),
) -> CoordinateDict:
    """
    Generate deterministic grid-cell centers for the urban demand area.

    Example:
        city_size_x = 40, city_size_y = 30, cell_size = 10
        -> 4 x 3 = 12 generation spots.
    """
    if city_size_x <= 0 or city_size_y <= 0:
        raise ValueError("city_size_x and city_size_y must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")
    if not math.isclose(city_size_x / cell_size, round(city_size_x / cell_size)) or not math.isclose(city_size_y / cell_size, round(city_size_y / cell_size)):
        raise ValueError("city_size_x and city_size_y must be divisible by cell_size.")

    n_cells_axis_x = int(round(city_size_x / cell_size))
    n_cells_axis_y = int(round(city_size_y / cell_size))
    half_x = city_size_x / 2.0
    half_y = city_size_y / 2.0
    cx, cy = center

    points: CoordinateDict = {}
    idx = 0

    for ix in range(n_cells_axis_x):
        for iy in range(n_cells_axis_y):
            x = cx - half_x + (ix + 0.5) * cell_size
            y = cy - half_y + (iy + 0.5) * cell_size
            points[idx] = (x, y)
            idx += 1

    return points

# Function to compute the number of generation points based on city size and cell size parameters
def compute_grid_generation_count(city_size_x: float, city_size_y: float, cell_size: float) -> int:
    if city_size_x <= 0 or city_size_y <= 0:
        raise ValueError("city_size_x and city_size_y must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")
    if not math.isclose(city_size_x / cell_size, round(city_size_x / cell_size)) or not math.isclose(city_size_y / cell_size, round(city_size_y / cell_size)):
        raise ValueError("city_size_x and city_size_y must be divisible by cell_size.")
    n_cells_axis_x = int(round(city_size_x / cell_size))
    n_cells_axis_y = int(round(city_size_y / cell_size))
    return n_cells_axis_x * n_cells_axis_y
#endregion

#region Transfer placement
# Random, but spatially disperesed. Divide the city into coarse blocks and place transfer stations approximately evenly across the area.
# For example, if S_max=8, place transfer stations in 8 randomly selected grid cells or in a roughly balanced layout.
def generate_transfer_stations(
    n_points: int,
    rng: random.Random,
    city_size_x: float,
    city_size_y: float,
    center: Point = (0.0, 0.0),
) -> CoordinateDict:
    """
    Generate random points in a city box, but distribute them more evenly
    than independent uniform sampling.

    The method creates a coarse grid with at least n_points cells, randomly
    selects n_points cells, and places one point randomly inside each selected cell.
    """
    if city_size_x <= 0 or city_size_y <= 0:
        raise ValueError("city_size_x and city_size_y must be positive.")
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    cx, cy = center
    half_x = city_size_x / 2.0
    half_y = city_size_y / 2.0

    aspect_ratio = city_size_x / city_size_y
    n_axis_x = math.ceil(math.sqrt(n_points * aspect_ratio))
    n_axis_y = math.ceil(n_points / n_axis_x)

    while n_axis_x * n_axis_y < n_points:
        if n_axis_x * n_axis_y < n_points:
            n_axis_y += 1
        if n_axis_x * n_axis_y < n_points:
            n_axis_x += 1

    coarse_cell_size_x = city_size_x / n_axis_x
    coarse_cell_size_y = city_size_y / n_axis_y

    candidate_cells = [
        (ix, iy)
        for ix in range(n_axis_x)
        for iy in range(n_axis_y)
    ]

    selected_cells = rng.sample(candidate_cells, n_points)

    points: CoordinateDict = {}

    for idx, (ix, iy) in enumerate(selected_cells):
        xmin = cx - half_x + ix * coarse_cell_size_x
        xmax = xmin + coarse_cell_size_x
        ymin = cy - half_y + iy * coarse_cell_size_y
        ymax = ymin + coarse_cell_size_y

        points[idx] = (
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax),
        )

    return points
#endregion

#region Place facilities
# Function to place Incinerators, landfills, and cement plants
def place_points_in_ring(
    indices: range,
    rng: random.Random,
    radius_min: float,
    radius_max: float,
    radius_center: float | None = None,   # optional argument for triangular distribution to place more points around a certain radius (e.g., for landfills around 50 km and cement plants around 200 km)
    center: Point = (0.0, 0.0),
    angle_min: float = 0.0,
    angle_max: float = 2.0 * math.pi,
) -> CoordinateDict:
    """
    Place points randomly in a circular ring around a center.

    If radius_center is None, radial distances are sampled uniformly.
    If radius_center is provided, radial distances are sampled from a
    triangular distribution with mode radius_center.

    Notes
    -----
    - The radius distribution controls distance from the city center.
    - The angle interval controls geographical direction.
    """
    if radius_min < 0:
        raise ValueError("radius_min must be non-negative.")

    if radius_min > radius_max:
        raise ValueError("radius_min must be smaller than or equal to radius_max.")

    if radius_center is not None:
        if not radius_min <= radius_center <= radius_max:
            raise ValueError("radius_center must lie between radius_min and radius_max.")

    if angle_min > angle_max:
        raise ValueError("angle_min must be smaller than or equal to angle_max.")
    
    cx, cy = center
    coordinates: CoordinateDict = {}

    for idx in indices:
        if radius_center is None:
            radius = rng.uniform(radius_min, radius_max)
        else:
            radius = rng.triangular(radius_min, radius_max, radius_center)

        angle = rng.uniform(angle_min, angle_max)

        coordinates[idx] = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )

    return coordinates
#endregion

#region Generate network
# Function to generate synthetic network locations for all facility types
def generate_grid_based_network_locations(
    G: range,
    S: range,
    I: range,
    L: range,
    C: range,
    rng: random.Random,
    city_size_x: float = 30.0,
    city_size_y: float = 30.0,
    grid_cell_size: float = 10.0,
    incinerator_radius_min: float = 10.0,
    incinerator_radius_max: float = 60.0,
    incinerator_radius_center: float = 35.0,   # place more incinerators around 35 km from city center
    landfill_radius_min: float = 40.0,
    landfill_radius_max: float = 120.0,
    landfill_radius_center: float = 75.0,   # place more landfills around 75 km from city center
    cement_radius_min: float = 80.0,
    cement_radius_max: float = 320.0,
    cement_radius_center: float = 200.0,    # place more cement plants around 200 km from city center
    incinerator_colocation_probability: float = 0.025,
    center: Point = (0.0, 0.0),
) -> Dict[str, CoordinateDict]:
    """
    Generate synthetic network locations with fixed grid-based generation spots.

    Generation spots are deterministic cell centers in a city grid.
    Transfer stations are placed randomly but evenly across the city.
    Incinerators, landfills, and cement plants are placed in rings around the city center, with some probability of co-location for incinerators at transfer stations.
    """

    waste_generation_points = generate_waste_generation_points(
        city_size_x=city_size_x,
        city_size_y=city_size_y,
        cell_size=grid_cell_size,
        center=center,
    )

    if len(G) > len(waste_generation_points):
        raise ValueError(
            f"len(G)={len(G)} exceeds available grid cells={len(waste_generation_points)}."
        )
    # else:
    #     print(f"Generated {len(waste_generation_points)} waste generation points based on city size and grid cell size, with {len(G)} used for the instance.")

    G_coords = {
        g: waste_generation_points[g]
        for g in G
    }

    S_coords = generate_transfer_stations(
        n_points=len(S),
        rng=rng,
        city_size_x=city_size_x,
        city_size_y=city_size_y,
        center=center,
    )

    I_coords: CoordinateDict = {}

    for i in I:
        if rng.random() < incinerator_colocation_probability:       # co-locate some incinerators at transfer stations for realism
            assigned_s = rng.choice(list(S))
            I_coords[i] = S_coords[assigned_s]
        else:
            radius = rng.triangular(incinerator_radius_min, incinerator_radius_max, incinerator_radius_center)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            I_coords[i] = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )

    L_coords = place_points_in_ring(
        indices=L,
        rng=rng,
        radius_min=landfill_radius_min,
        radius_max=landfill_radius_max,
        radius_center=landfill_radius_center,
        center=center,
    )

    C_coords = place_points_in_ring(
        indices=C,
        rng=rng,
        radius_min=cement_radius_min,
        radius_max=cement_radius_max,
        radius_center=cement_radius_center,
        center=center,
    )

    return {
        "G": G_coords,
        "S": S_coords,
        "I": I_coords,
        "L": L_coords,
        "C": C_coords,
    }
#endregion

#region Euclidean distance
def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
#endregion

#region Road distance
# Function to compute road distance based on network structure
def road_distance(
    origin: Point,
    destination: Point,
    rng: random.Random,
    detour_min: float,
    detour_max: float,
    noise_share: float,
) -> int:
    """
    Compute synthetic road distance between two coordinates.

    Coordinates are interpreted as kilometres. The distance is computed as
    Euclidean distance multiplied by a random road-detour factor and perturbed
    by small relative noise.

    No clipping is applied. Therefore, all distance realism should be induced
    by the coordinate-generation procedure.
    """
    euclidean = euclidean_distance(origin, destination)

    # Only exact or numerical co-location should yield distance zero.
    if math.isclose(euclidean, 0.0, abs_tol=1e-1):
        return 0

    detour_factor = rng.uniform(detour_min, detour_max)
    noise = rng.uniform(-noise_share, noise_share) * euclidean

    distance = detour_factor * euclidean + noise
    rounded_distance = int(round(distance))

    # Prevent non-co-located but very close facilities from becoming zero.
    return max(1, rounded_distance)
# Function to compute road distance with clipping (i.e. enforce transport distance bounds) and noise
def clipped_road_distance(
    origin: Point,
    destination: Point,
    rng: random.Random,
    min_distance: float,
    max_distance: float,
    detour_min: float,
    detour_max: float,
    noise_share: float,
) -> int:
    euclidean = euclidean_distance(origin, destination)

    if math.isclose(euclidean, 0.0, abs_tol=1e-1):
        return 0

    detour_factor = rng.uniform(detour_min, detour_max)
    noise = rng.uniform(-noise_share, noise_share) * euclidean

    distance = detour_factor * euclidean + noise
    distance = max(min_distance, min(max_distance, distance))

    rounded_distance = int(round(distance))

    if min_distance == 0.0 and rounded_distance == 0:
        return 1

    return rounded_distance
# endregion

#region Transport distance
# Function to build transport distance matrix from origins to destinations based on their coordinates and the road distance function
def build_transport_distance_matrix(
    origins: CoordinateDict,
    destinations: CoordinateDict,
    rng: random.Random,
    min_distance: float,
    max_distance: float,
    detour_min: float,
    detour_max: float,
    noise_share: float,
    clipping: bool = False,
) -> List[List[int]]:
    origin_ids = sorted(origins)
    destination_ids = sorted(destinations)

    if clipping:
        return [
            [
                clipped_road_distance(
                    origin=origins[o],
                    destination=destinations[d],
                    rng=rng,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    detour_min=detour_min,
                    detour_max=detour_max,
                    noise_share=noise_share,
                )
                for d in destination_ids
            ]
            for o in origin_ids
        ]
    else:
        return [
            [
                road_distance(
                    origin=origins[o],
                    destination=destinations[d],
                    rng=rng,
                    detour_min=detour_min,
                    detour_max=detour_max,
                    noise_share=noise_share,
                )
                for d in destination_ids
            ]
            for o in origin_ids
        ]
#endregion

#region Capacity Management
def _raise_absolute_capacities_to_target_random(
    capacities: list[int],
    capacity_classes: list[int],
    target_total_capacity: int,
    rng: random.Random,
) -> list[int]:
    """
    Randomly upgrade facilities to the next larger capacity class until the
    total capacity reaches at least target_total_capacity.

    In each iteration, one facility is selected uniformly at random among all
    facilities that can still be upgraded. This preserves more of the initially
    sampled random capacity structure than always upgrading the smallest facility.
    """
    capacities = capacities.copy()
    capacity_classes = sorted(set(capacity_classes))

    while sum(capacities) < target_total_capacity:
        upgrade_candidates = []

        for idx, current_capacity in enumerate(capacities):
            larger_classes = [
                capacity_class
                for capacity_class in capacity_classes
                if capacity_class > current_capacity
            ]

            if not larger_classes:
                continue

            next_capacity = min(larger_classes)

            upgrade_candidates.append((idx, next_capacity))

        if not upgrade_candidates:
            raise RuntimeError(
                "Capacity target could not be reached although feasibility was "
                "expected. Check capacity_classes and n_facilities."
            )

        selected_idx, next_capacity = rng.choice(upgrade_candidates)
        capacities[selected_idx] = next_capacity

    return capacities

def _raise_lowest_absolute_capacities_to_target(
    capacities: list[int],
    capacity_classes: list[int],
    target_total_capacity: int,
) -> list[int]:
    """
    Upgrade facilities to the next larger capacity class until the total
    capacity reaches at least target_total_capacity.

    The upgrade rule prioritizes the currently smallest facility that can still
    be upgraded. This preserves the discrete capacity structure and avoids
    arbitrary continuous correction.
    """
    capacities = capacities.copy()
    capacity_classes = sorted(set(capacity_classes))

    while sum(capacities) < target_total_capacity:
        upgrade_candidates = []

        for idx, current_capacity in enumerate(capacities):
            larger_classes = [
                capacity_class
                for capacity_class in capacity_classes
                if capacity_class > current_capacity
            ]

            if not larger_classes:
                continue

            next_capacity = min(larger_classes)
            increase = next_capacity - current_capacity

            upgrade_candidates.append(
                (current_capacity, increase, idx, next_capacity)
            )

        if not upgrade_candidates:
            raise RuntimeError(
                "Capacity target could not be reached although feasibility was "
                "expected. Check capacity_classes and n_facilities."
            )

        # Upgrade one of the lowest-capacity facilities first.
        # Tie-breaker: smallest increase.
        _, _, selected_idx, next_capacity = min(upgrade_candidates)

        capacities[selected_idx] = next_capacity

    return capacities
#endregion

#region Facility capacities
def allocate_absolute_capacity_classes(
    target_total_capacity: int,
    n_facilities: int,
    rng: random.Random,
    capacity_classes: list[int],
    class_probabilities: list[float] | None = None,
    enforce_total: bool = True,
    upgrade_rule: Literal["random", "lowest"] = "random"
) -> list[int]:
    """
    Allocate facility capacities from predefined absolute capacity classes.

    The function first samples one capacity class per facility. If the sampled
    total capacity is below the required target and enforce_total=True, it
    iteratively upgrades one of the facilities (lowest or random) to the next
    larger class until the target total capacity is reached.

    Parameters
    ----------
    target_total_capacity:
        Required total capacity across all facilities, e.g. t/year.

    n_facilities:
        Number of facilities.

    rng:
        Random number generator.

    capacity_classes:
        Absolute capacity classes, e.g. [500_000, 750_000, 1_000_000].

    class_probabilities:
        Sampling probabilities for the capacity classes. If None, all classes
        are sampled with equal probability.

    enforce_total:
        If True, sampled capacities are upgraded until their sum is at least
        target_total_capacity.

    Returns
    -------
    list[int]
        Capacity assigned to each facility.
    """
    if target_total_capacity <= 0:
        raise ValueError("target_total_capacity must be positive.")

    if n_facilities <= 0:
        raise ValueError("n_facilities must be positive.")

    if not capacity_classes:
        raise ValueError("capacity_classes must not be empty.")

    if any(capacity <= 0 for capacity in capacity_classes):
        raise ValueError("All capacity classes must be positive.")

    capacity_classes = sorted(set(capacity_classes))

    if class_probabilities is not None:
        if len(class_probabilities) != len(capacity_classes):
            raise ValueError(
                "class_probabilities must have the same length as capacity_classes."
            )

        if any(probability < 0 for probability in class_probabilities):
            raise ValueError("class_probabilities must be non-negative.")

        probability_sum = sum(class_probabilities)

        if probability_sum <= 0:
            raise ValueError("At least one class probability must be positive.")

        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("class_probabilities must sum to 1.")

    maximum_possible_capacity = n_facilities * max(capacity_classes)

    if enforce_total and maximum_possible_capacity < target_total_capacity:
        raise ValueError(
            "Target total capacity cannot be reached with the given number of "
            f"facilities and capacity classes. Required: {target_total_capacity:,}; "
            f"maximum possible: {maximum_possible_capacity:,}."
        )

    capacities = rng.choices(
        population=capacity_classes,
        weights=class_probabilities,
        k=n_facilities,
    )

    if enforce_total:
        if upgrade_rule == "random":
            capacities = _raise_absolute_capacities_to_target_random(
                capacities=capacities,
                capacity_classes=capacity_classes,
                target_total_capacity=target_total_capacity,
                rng=rng,
            )
        elif upgrade_rule == "lowest":
            capacities = _raise_lowest_absolute_capacities_to_target(
                capacities=capacities,
                capacity_classes=capacity_classes,
                target_total_capacity=target_total_capacity,
            )
        

    return capacities
#endregion

#region Waste Generation
def sample_total_waste_generation(
    rng,
    city_size_x: float,
    city_size_y: float,
    waste_density_t_per_km2_year: float = 3000.0,
    lower_factor: float = 0.8,
    upper_factor: float = 1.4,
    rounding_base: int = 1000,
) -> float:
    """
    Sample annual MSW generation for a synthetic Chinese mega-city core.

    Parameters
    ----------
    city_size_km:
        Side length of the square study area in km.
    waste_density_t_per_km2_year:
        Effective annual MSW generation intensity.
    lower_factor:
        Lower multiplier around the reference generation.
    upper_factor:
        Upper multiplier around the reference generation.

    Returns
    -------
    float
        Annual MSW generation in tonnes/year.
    """
    city_area = city_size_x * city_size_y
    mean_gen = waste_density_t_per_km2_year * city_area
    min_gen = lower_factor * mean_gen
    max_gen = upper_factor * mean_gen

    sample_gen_value = rng.triangular(min_gen, max_gen, mean_gen)

    return int(round(sample_gen_value / rounding_base) * rounding_base)
#endregion

#region Validate System Capa
def validate_system_capacity(
    Q_gen_total: int,
    Q_s: list[int],
    Q_i: list[int],
    Q_l: list[int],
    Q_k: list[int],
    C_total: int,
    kappa_land: float,
    kappa_coproc: float,
) -> None:
    total_transfer_capacity = sum(Q_s)
    total_incineration_capacity = sum(Q_i)
    total_landfill_capacity = sum(Q_l)
    total_coprocessing_capacity = max(Q_k) * C_total

    if total_transfer_capacity < Q_gen_total:
        raise ValueError(
            f"Insufficient transfer capacity: {total_transfer_capacity:,} "
            f"< Q_gen_total={Q_gen_total:,}."
        )

    effective_landfill_capacity = min(
        total_landfill_capacity,
        kappa_land * Q_gen_total,
    )

    effective_coprocessing_capacity = min(
        total_coprocessing_capacity,
        kappa_coproc * Q_gen_total,
    )

    effective_disposal_capacity = (
        total_incineration_capacity
        + effective_landfill_capacity
        + effective_coprocessing_capacity
    )

    if effective_disposal_capacity < Q_gen_total:
        raise ValueError(
            "Insufficient aggregate disposal capacity after policy limits: "
            f"{effective_disposal_capacity:,.0f} < {Q_gen_total:,}."
        )
    else:
        print(
            f"Validated system capacity:\n"
            f"Total waste generation={Q_gen_total:,}\n"
            f"Total transfer capacity={total_transfer_capacity:,}\n"
            f"Total incineration capacity={total_incineration_capacity:,}\n"
            f"Total landfill capacity={total_landfill_capacity:,} (effective {effective_landfill_capacity:,.0f} with kappa_land={kappa_land})\n"
            f"Total co-processing capacity={total_coprocessing_capacity:,} (effective {effective_coprocessing_capacity:,.0f} with kappa_coproc={kappa_coproc}), \n"
            f"Aggregate effective disposal capacity={effective_disposal_capacity:,.0f}"
        )
#endregion

#endregion

####################################################################################
############################### Data definition ####################################
####################################################################################

#region Instance generation
# stylized, empirically calibrated Chinese megacity systems
def generate_instance(
        S_total: int,
        I_total: int,
        L_total: int,
        C_total: int,
        city_size_x: float = 30.0,
        city_size_y: float = 30.0,
        grid_cell_size: float = 10.0,
        waste_gen_density: int = 3000,     # effective annual waste intensity in chinese mega cities 2500-4000 t/km² per year
        incinerator_radius_min: float = 10.0,
        incinerator_radius_max: float = 60.0,
        incinerator_radius_center: float = 35.0,   # place more incinerators around 35 km from city center
        incinerator_colocation_probability: float = 0.025,
        landfill_radius_min: float = 40.0,
        landfill_radius_max: float = 120.0,
        landfill_radius_center: float = 75.0,   # place more landfills around 75 km from city center
        cement_radius_min: float = 80.0,
        cement_radius_max: float = 320.0,
        cement_radius_center: float = 200.0,    # place more cement plants around 200 km from city center
        clipping_distances: bool = False,
        budget_mun_availability: float = 0.8,   # Available subsidy budget as a fraction of the maximum required budget (all waste at maximum subsidy)
        budget_cem_availability: float = 0.65,   # Available investment budget as a fraction of the maximum required budget (all kilns at maximum capacity)
        seed: int = 7,
) -> InstanceData:
    rng = random.Random(seed)

    # -----------------------------
    # SETS
    # -----------------------------
    # G_max = 8
    # S_max = 8
    # I_max = 6
    # L_max = 3
    # C_max = 6
    
    W_max = 2
    K_max = 3
    F_max = 1   # necessary to choose diferent coal types?
    H_max = 5

    # G = range(G_max)
    S = range(S_total)
    I = range(I_total)
    L = range(L_total)
    C = range(C_total)
    
    W = range(W_max)
    K = range(K_max)
    F = range(F_max)
    H = range(H_max)

    # "Anhui Conch Cement (cluster)", "Suzhou Dahua Marine", "Jiangsu Pengfei (Haian)", "Zhejiang Producer A", "Jiangsu Producer A","Anhui Producer A"
    cement_names = [f"Cement Plant {i+1}" for i in C]  # Placeholder names; replace with actual names if desired

    G_max = compute_grid_generation_count(
        city_size_x=city_size_x, 
        city_size_y=city_size_y, 
        cell_size=grid_cell_size
    )
    G = range(G_max)

    # -----------------------------
    # DISTANCES (km) - synthetic but plausible
    # generation -> transfer: 2..50 km
    # transfer -> incinerator: 0..60 km (co-location possible)
    # transfer -> landfill (neighbour districts): 30..120 km
    # transfer -> cement (Jiangsu/Anhui/Zhejiang): 80..320 km
    # -----------------------------
    network_locations = generate_grid_based_network_locations(
        G=G,
        S=S,
        I=I,
        L=L,
        C=C,
        rng=rng,
        city_size_x=city_size_x,
        city_size_y=city_size_y,
        grid_cell_size=grid_cell_size,
        incinerator_radius_min=incinerator_radius_min,
        incinerator_radius_max=incinerator_radius_max,
        incinerator_radius_center=incinerator_radius_center,
        landfill_radius_min=landfill_radius_min,
        landfill_radius_max=landfill_radius_max,
        landfill_radius_center=landfill_radius_center,
        cement_radius_min=cement_radius_min,
        cement_radius_max=cement_radius_max,
        cement_radius_center=cement_radius_center,
        incinerator_colocation_probability=incinerator_colocation_probability,
        center=(0.0, 0.0),
    )

    G_coords = network_locations["G"]
    S_coords = network_locations["S"]
    I_coords = network_locations["I"]
    L_coords = network_locations["L"]
    C_coords = network_locations["C"]

    TD_gs = build_transport_distance_matrix(
        origins=G_coords,
        destinations=S_coords,
        rng=rng,
        min_distance=2.0,
        max_distance=50.0,
        detour_min=1.05,
        detour_max=1.20,
        noise_share=0.02,
        clipping=clipping_distances,
    )

    TD_si = build_transport_distance_matrix(
        origins=S_coords,
        destinations=I_coords,
        rng=rng,
        min_distance=0.0,
        max_distance=60.0,
        detour_min=1.05,
        detour_max=1.30,
        noise_share=0.04,
        clipping=clipping_distances,
    )

    TD_si_avg = sum(TD_si[s][i] for s in S for i in I) / (S_total * I_total)
    
    TD_sl = build_transport_distance_matrix(
        origins=S_coords,
        destinations=L_coords,
        rng=rng,
        min_distance=30.0,
        max_distance=120.0,
        detour_min=1.10,
        detour_max=1.40,
        noise_share=0.04,
        clipping=clipping_distances,
    )

    TD_sc = build_transport_distance_matrix(
        origins=S_coords,
        destinations=C_coords,
        rng=rng,
        min_distance=80.0,
        max_distance=320.0,
        detour_min=1.10,
        detour_max=1.40,
        noise_share=0.04,
        clipping=clipping_distances,
    )

    # -----------------------------
    # EMISSIONS / COSTS
    # w=0: high moisture/chlorine, most organic and low fossil plastic content (worse)
    # w=1: RDF-like medium moisture/chlorine, higher fossil plastic content (better)
    # -----------------------------
    epsilon_truck = 0.0002          # tCO2e/t-km, refuse truck (heavy duty long haul truck 0.00006 tCO2e/t-km, but add empty return trips, i.e. double emissions per ton transported)
    epsilon_land = [0.85, 0.60]
    epsilon_inc = [0.55, 0.40]
    epsilon_kiln_w = [0.22, 0.68]
    epsilon_kiln_f = [2.25]   # epsilon_kiln_f = [2.25, 2.59]

    c_truck = 0.45      # CNY/t-km
    c_land = 180.0       # CNY/t
    c_inc = 200.0       # CNY/t

    # -----------------------------
    # WASTE GENERATION (t/year):
    # Total MSW in a synthetic chinese Mega-city
    # E.g. Shanghai generates around 10,000,000–17,000,000 t/year in total
    # Split by type: 40-70% high moisture, rest medium moisture.
    # -----------------------------
    # total_target = rng.randint(6_000_000, 9_000_000)
    total_target = sample_total_waste_generation(
        rng=rng, 
        city_size_x=city_size_x, 
        city_size_y=city_size_y, 
        waste_density_t_per_km2_year=waste_gen_density
    )
    # split = [0.55, 0.45]      # fixed split
    # distribute by district (G) using a Dirichlet-like random split
    weights = [rng.random() for _ in G]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    Q_gw = []
    for g in G:
        g_total = int(round(total_target * weights[g]))
        split_w0 = rng.uniform(0.4, 0.7)
        split = [split_w0, 1 - split_w0]
        
        q0 = int(round(g_total * split[0]))
        q1 = max(0, g_total - q0)
        Q_gw.append([q0, q1])

    # Forced rounding correction to ensure total generation matches target after rounding to integers:
    current_total = sum(Q_gw[g][w] for g in G for w in W)
    difference = total_target - current_total
    if difference != 0:
        g = rng.choice(list(G))
        w = rng.choice(list(W))

        Q_gw[g][w] += difference    # adjust a random cell to fix any rounding-induced discrepancy, ensuring total generation matches the target

    Q_gen_total = sum(Q_gw[g][w] for g in G for w in W)
    total_Q_gen_per_w = [sum(Q_gw[g][w] for g in G) for w in W]

    # Transfer station capacity
    # ensure > inbound per station; keep loose
    # If each district maps mostly to one transfer, set capacity between 1000..4100 t/day (capacity classes)
    transfer_capacity_factor = rng.triangular(1.05, 1.25, 1.15)
    Q_s_total = int(round(Q_gen_total * transfer_capacity_factor))

    Q_s_classes = [182_500, 365_000, 500_000, 750_000, 1_000_000, 1_500_000]     # t/year, corresponds to [500, 1000, 1370, 2055, 2740, 4110] t/day

    Q_s = allocate_absolute_capacity_classes(
        target_total_capacity=Q_s_total,
        n_facilities=S_total,
        rng=rng,
        capacity_classes=Q_s_classes,
        class_probabilities=[0.31, 0.40, 0.20, 0.03, 0.03, 0.03],
        enforce_total=True,
        upgrade_rule="random",
    )

    # Incineration and landfill capacities:
    # Set so that inc+land can cover all waste (to avoid forced investment), but landfill quota still restricts landfill share.
    # Q_i = [int(round(rng.triangular(700, 1200, 950))) for _ in I]  # 6 plants
    # Q_l = [int(round(rng.triangular(1000, 1300, 1200))) for _ in L]  # 3 sites

    incineration_capacity_factor = rng.triangular(0.75, 0.95, 0.85)
    Q_i_total = int(round(Q_gen_total * incineration_capacity_factor))

    Q_i_classes = [365_000, 550_000, 750_000, 1_100_000, 1_850_000]     # t/year, corresponds to [1000, 1507, 2055, 3014, 5068] t/day

    Q_i = allocate_absolute_capacity_classes(
        target_total_capacity=Q_i_total,
        n_facilities=I_total,
        rng=rng,
        capacity_classes=Q_i_classes,
        class_probabilities=[0.10, 0.25, 0.35, 0.25, 0.05],
        enforce_total=True,
        upgrade_rule="random",
    )

    landfill_capacity_factor = rng.triangular(0.35, 0.50, 0.42)
    Q_l_total = int(round(Q_gen_total * landfill_capacity_factor))

    Q_l_classes = [1_000_000, 2_000_000, 3_000_000]         # t/year, corresponds to [2739, 5479, 8218] t/day

    Q_l = allocate_absolute_capacity_classes(
        target_total_capacity=Q_l_total,
        n_facilities=L_total,
        rng=rng,
        capacity_classes=Q_l_classes,
        class_probabilities=[0.20, 0.45, 0.35],
        enforce_total=True,
        upgrade_rule="random",
    )

    # -----------------------------
    # CO-PROCESSING OPTIONS (t/year)
    # -----------------------------
    Q_k_daily = [350, 500, 1000]
    Q_k = [q*365 for q in Q_k_daily]
    Q_k_max = max(Q_k)

    # -----------------------------
    # POLICY / WEIGHTS
    # -----------------------------
    kappa_land = 0.35
    kappa_coproc = 0.40

    validate_system_capacity(
        Q_gen_total=Q_gen_total,
        Q_s=Q_s,
        Q_i=Q_i,
        Q_l=Q_l,
        Q_k=Q_k,
        C_total=C_total,
        kappa_land=kappa_land,
        kappa_coproc=kappa_coproc,
    )

    phi_max = [220.0, 175.0]  # [high moisture, medium moisture]
    phi_wh = [[(h / (H_max - 1)) * phi_max[w] for h in H] for w in W]

    # U_w = [min(sum(Q_gw[g][w] for g in G), Q_k_max*len(C)) for w in W]  # Upper bound on waste flow of type w (can be tightened based on data)
    # A waste type w cannot flow trough network in an amount larger than: (i) total generated amount, (ii) total transfer-station capacity, (iii) total co-processing capacity
    U_w = [min(total_Q_gen_per_w[w], sum(Q_s), Q_k_max*len(C)) for w in W]  # Upper bound on waste flow of type w (can be tightened based on data)

    # Only 'budget_mun_availability' % of the maximum total potential waste flow to kilns can be subsidized supposing maximum subsidy levels, 
    # to create a more realistic budget constraint that requires trade-offs in subsidy allocation
    budget_municipality = budget_mun_availability * sum(phi_max[w] * U_w[w] for w in W)  # Set municipal budget based on maximum potential subsidy payout with some availability factor

    # -----------------------------
    # FOLLOWER: coal types, costs, kiln demands
    # -----------------------------
    # Coal mixtures: e.g., lower-grade and higher-grade thermal coal
    price_f = [700.0]   # CNY/t      # price_f = [700.0, 850.0]
    beta_f = [23.0]      # GJ/t (two mixes)       # beta_f = [23.0, 27.0]

    # Kiln daily energy requirement
    # 2.500-5.000 t clinker / day (sometimes up to 10.000 t/day possible), 3 - 3.7 GJ/t clinker
    # daily energy requirement per plant: 7,500 - 18,500 GJ/day
    alpha_c_daily = [rng.triangular(7000, 18000, 15000) for _ in C]
    alpha_c = [alpha * 365 for alpha in alpha_c_daily]

    beta_w = [12.0*0.4, 16.0*0.7]   # GJ/t for waste types, adjusted by moisture content reduction in pre-processing (high moisture reduced by 60%, medium moisture reduced by 30%)

    # Investment CAPEX by option size (CNY) – extend to K=3
    c_invest_k = [90_000_000, 120_000_000, 300_000_000]

    # Preprocessing cost by waste type
    c_preproc_w = [150.0, 125.0]

    c_penalty = 100.0

    # Only 'budget_cem_availability' % of the maximum total potential investment cost for co-processing is available
    budget_cem = budget_cem_availability * C_total * max(c_invest_k)  # bigger portfolio-level budget for 6 plants

    tau = 1e-3

    # Levelized daily fixed cost per option k
    i_rate = 0.0325
    lifetime_years = 15
    CRF = crf(i_rate, lifetime_years)
    capex_ann = [c_invest_k[k] * CRF for k in K]
    opex_fix_ann = [c_invest_k[k] * 0.06 for k in K]
    fixcost_invest_unscaled_k = [capex_ann[k] + opex_fix_ann[k] for k in K]
    fixcost_invest_k = [cost for cost in fixcost_invest_unscaled_k]             # Annualized fixed investment-equivalent cost (CNY/year)
    # fixcost_invest_k = [cost/1000 for cost in fixcost_invest_unscaled_k]     # divide by 1000 to scale down to daily cost, because only 0.1% of annual waste is modeled in this instance

    # Big-M value for cut generation
    M_primal = {
        'F3': 1,
        # 'F4': max(alpha_c)*kappa_coproc+10,     # Maximmum energy content in co-processing
        'F4': {c: (alpha_c[c]*kappa_coproc) + 1 for c in C},     # Maximmum energy content in co-processing
        'F5': Q_k_max+1,                        # Maximum co-processing quantity (not really needed, because x_ck_fixed is already fixed in the OC block, thus the maximal capacity is deterministic based on the fixed investment decision; keep it for fallback)
        'F6': {s: {w: float(min(sum(Q_gw[g][w] for g in G), Q_s[s])) for w in W} for s in S},  # Maximum waste flow from transfer station s to kiln c based on total generation and station capacity
        # 'q_cf': max(alpha_c)+1,                 # Maximum quantity of coal processed at cement plant (based on maximum energy content needed)
        # since alpha_c is in GJ and beta_f is in GJ/t, a physically meaningful coal bound is closer to alpha_c[c] / beta_f[f] + 1.0
        'q_cf': {c: {f: alpha_c[c] / beta_f[f] + 1 for f in F} for c in C},   # Maximum quantity of coal processed at cement plant (based on maximum energy content needed)
        # 'q_scw': Q_k_max+1,                     # Maximum quantity of waste allocated from transfer station to cement plant
        'q_scw': {s: {c: {w: float(min(Q_s[s], Q_k_max, total_Q_gen_per_w[w]))+1 for w in W} for c in C} for s in S},  # Maximum quantity of waste allocated from transfer station to cement plant
        # 'r_sw': max(Q_s)+1,                   # Maximum residual waste at transfer station after allocation
        # 'r_sw': {s: int(Q_s[s])+1 for s in S},  # Maximum residual waste at transfer station after allocation, capcitated by individual capacities of transfer stations
        'r_sw': {s: {w: float(min(Q_s[s], total_Q_gen_per_w[w]))+1 for w in W} for s in S},  # Maximum residual waste at transfer station after allocation, capcitated by individual capacities of transfer stations
    }

    M_dual = {
        # 'lam_F3': 1e3,     # Big-M for dual variable of constraint F3 (energy fulfillment constraint)
        # p_{f}-\lambda^{F3}_c\beta_f-\pi^{1}_{cf} = 0 with \lambda^{F3}_c >= 0 and \pi^{1}_{cf} >= 0; rearrange to \pi^{1}_{cf} = p_{f}-\lambda^{F3}_c\beta_f; it follows p_{f}-\lambda^{F3}_c\beta_f >= 0 and thus \lambda^{F3}_c <= p_{f}/\beta_f for all f; so a reasonable Big-M for \lambda^{F3}_c is max(p_{f}/\beta_f) + 1 to allow for some numerical tolerance
        'lam_F3': min(price_f[f] / beta_f[f] for f in F) + 1,     # Big-M for dual variable of constraint F3 (energy fulfillment constraint)
        'lam_F4': 1e4,     # Big-M for dual variable of constraint F4 (maximum co-processing quantity)
        'lam_F5': 1e4,     # Big-M for dual variable of constraint F5 (co-process capacity limited by investment decision)
        'lam_F6': 1e4,     # Big-M for dual variable of constraint F6 (waste flow from transfer station to kiln limited by generation and station capacity)
        # derived from stationarity for q_cf: data.price_f[f] - lam_F3[c]*data.beta_f[f] - pi_q_cf[c,f] == 0 with lam_F3 >= 0 and beta_f >= 8, so price_f is a reasonable upper bound for pi_q_cf
        'pi_q_cf': max(price_f)+1,    # Big-M for dual variable of constraint limiting quantity of coal processed at cement plant
        'pi_q_scw': 1e4,   # Big-M for dual variable of constraint limiting quantity of waste allocated from transfer station to cement plant
        'pi_r_sw': 1e4,    # Big-M for dual variable of constraint limiting residual waste at transfer station after allocation
    }

    return InstanceData(
        G_max=G_max, S_max=S_total, I_max=I_total, L_max=L_total, C_max=C_total,
        W_max=W_max, K_max=K_max, F_max=F_max, H_max=H_max,
        G=G, S=S, W=W, I=I, L=L, C=C, K=K, F=F, H=H,

        G_coords=G_coords, S_coords=S_coords, I_coords=I_coords, L_coords=L_coords, C_coords=C_coords,

        cement_names=cement_names,

        TD_gs=TD_gs, TD_sl=TD_sl, TD_si=TD_si, TD_si_avg=TD_si_avg, TD_sc=TD_sc,

        epsilon_truck=epsilon_truck,
        epsilon_land=epsilon_land, epsilon_inc=epsilon_inc, epsilon_kiln_w=epsilon_kiln_w, epsilon_kiln_f=epsilon_kiln_f,

        c_truck=c_truck, c_land=c_land, c_inc=c_inc,
        
        Q_gw=Q_gw, Q_gen_total=Q_gen_total,
        Q_s=Q_s, Q_l=Q_l, Q_i=Q_i,
        Q_k=Q_k, Q_k_max=Q_k_max,
        
        kappa_land=kappa_land, kappa_coproc=kappa_coproc,
        budget_municipality=budget_municipality,
        phi_max=phi_max,
        phi_wh=phi_wh,
        price_f=price_f, beta_f=beta_f,
        alpha_c=alpha_c, beta_w=beta_w,
        c_invest_k=c_invest_k, c_preproc_w=c_preproc_w,
        c_penalty=c_penalty, budget_cem=budget_cem,
        tau=tau,
        fixcost_invest_k=fixcost_invest_k,
        M_primal=M_primal, M_dual=M_dual,
        U_w=U_w
    )
#endregion

__all__ = ["InstanceData", "generate_instance"]

if __name__ == "__main__":
    instance = generate_instance(
        S_total=8,
        I_total=6,
        L_total=3,
        C_total=6,

        city_size_x=30.0,
        city_size_y=30.0,
        grid_cell_size=10.0,
        waste_gen_density=3000,

        incinerator_radius_min=10.0,
        incinerator_radius_max=60.0,
        incinerator_radius_center=35.0,
        landfill_radius_min=40.0,
        landfill_radius_max=120.0,
        landfill_radius_center=75.0,
        cement_radius_min=80.0,
        cement_radius_max=320.0,
        cement_radius_center=200.0,
        incinerator_colocation_probability=0.025,

        clipping_distances=False,
        seed=7,
    )
    print(instance)