from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import List, Dict, Tuple, Any, Literal

from matplotlib.pylab import add


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

#region Parameter data class
@dataclass(frozen=True )
class InstanceParameters:
    S_total: int
    I_total: int
    L_total: int
    C_total: int

    city_size_x: float = 30.0
    city_size_y: float = 30.0
    grid_cell_size: float = 10.0
    waste_gen_density: int = 3000                       # effective annual waste intensity in chinese mega cities 2500-4000 t/km² per year

    incinerator_radius_min: float = 10.0
    incinerator_radius_max: float = 60.0
    incinerator_radius_center: float = 35.0             # place more incinerators around 35 km from city center
    incinerator_colocation_probability: float = 0.05

    landfill_radius_min: float = 40.0
    landfill_radius_max: float = 120.0
    landfill_radius_center: float = 65.0                # place more landfills around 65 km from city center

    cement_radius_min: float = 80.0
    cement_radius_max: float = 250.0
    cement_radius_center: float = 150.0                 # place more cement plants around 150 km from city center

    clipping_distances: bool = False

    budget_mun_availability: float = 0.8                # Available subsidy budget as a fraction of the maximum required budget (all waste at maximum subsidy)
    budget_cem_availability: float = 0.65               # Available investment budget as a fraction of the maximum required budget (all kilns at maximum capacity)

    c_truck: float = 0.45                               # CNY/t-km
    c_land: float = 180.0                               # CNY/t
    c_inc: float = 200.0                                # CNY/t

    kappa_land: float = 0.30                            # Maximum allowed landfill quota/capacity
    kappa_coproc: float = 0.50                          # Maximum co-processing quota/capacity

    local_high_moisture_bounds_node: List[float] = field(default_factory=lambda: [0.4, 0.7])        # range for heterogeneous high moisture share per node
    high_moisture_share_total_network: float = 0.60                                                 # target high moisture share for the entire network (to ensure constant overall waste split for benchmark)

    phi_max_w: List[float] = field(default_factory=lambda: [220.0, 175.0])             # [high moisture, medium moisture]

    price_coal_f: List[float] = field(default_factory=lambda: [700.0])                 # CNY/t

    c_preproc_w: List[float] = field(default_factory=lambda: [150.0, 125.0])           # CNY/t for pre-processing (sorting, shredding, drying) of waste types (high moisture, medium moisture)
    c_penalty: float = 100.0                            # CNY/t penalty of denied allocated waste quota

    bigM_duals_unrestricted: float = 1e4                             # Big-M values for dual variables in KKT cuts, where no explicit UB can be derived
                                                        # Default placeholder for unrestricted dual-variable Big-M bounds. The actual value is an algorithmic 
                                                        # parameter and may be overwritten by run_yue_decomposition() at solve time.
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
    total_generation_target = waste_density_t_per_km2_year * city_area

    return total_generation_target

    # Compute mean, min, and max generation values based on the city area and waste density, then sample from a triangular distribution.

    # mean_gen = waste_density_t_per_km2_year * city_area
    # min_gen = lower_factor * mean_gen
    # max_gen = upper_factor * mean_gen

    # sample_gen_value = rng.triangular(min_gen, max_gen, mean_gen)

    # return int(round(sample_gen_value / rounding_base) * rounding_base)
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
    # effective_coprocessing_capacity = sum(
    #     min(
    #         max(Q_k),
    #         kappa_coproc * alpha_c[c] / min(beta_w)
    #     )
    #     for c in C
    # )

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

#region Waste Composition
def allocate_waste_composition(
    g_totals: list[int],
    rng: random.Random,
    overall_high_moisture_share: float,
    local_share_min: float,
    local_share_max: float,
) -> list[list[int]]:
    """
    Allocate each generation node's waste to two waste types while enforcing an exact aggregate high-moisture share.

    Local high-moisture shares are sampled randomly within [local_share_min, local_share_max] and subsequently adjusted
    such that their waste-weighted average equals overall_high_moisture_share.

    Parameters
    ----------
    g_totals:
        Total waste generation at each generation node.
    rng:
        Random number generator.
    overall_high_moisture_share:
        Target system-wide share of high-moisture waste.
    local_share_min:
        Minimum local high-moisture share.
    local_share_max:
        Maximum local high-moisture share.

    Returns
    -------
    list[list[int]]
        Waste quantities [high_moisture, medium_moisture] for each node.
    """
    if not 0.0 <= local_share_min <= local_share_max <= 1.0:
        raise ValueError("Local waste-share bounds must lie within [0, 1].")

    if not local_share_min <= overall_high_moisture_share <= local_share_max:
        raise ValueError(
            "Overall high-moisture share must lie within the local share bounds."
        )

    total_waste = sum(g_totals)

    if total_waste <= 0:
        raise ValueError("Total waste generation must be positive.")

    # Integer aggregate target. This is the closest feasible integer quantity to the requested system-wide composition.
    target_high_moisture = int(round(overall_high_moisture_share * total_waste))

    effective_target_share = target_high_moisture / total_waste

    # 1. Draw heterogeneous local shares.
    raw_shares = [rng.uniform(local_share_min, local_share_max) for _ in g_totals]

    raw_weighted_share = (sum(q * share for q, share in zip(g_totals, raw_shares)) / total_waste)

    # 2. Adjust the random shares while preserving their ordering and bounds.
    if math.isclose(
        raw_weighted_share,
        effective_target_share,
        rel_tol=1e-12,              # loser bound for some variance, e.g. 0.6 determined but 0.59-0.61 is acceptable?
        abs_tol=1e-12,
    ):
        adjusted_shares = raw_shares

    elif raw_weighted_share < effective_target_share:
        alpha = (
            (effective_target_share - raw_weighted_share)
            / (local_share_max - raw_weighted_share)
        )

        adjusted_shares = [
            share + alpha * (local_share_max - share)
            for share in raw_shares
        ]

    else:
        alpha = (
            (raw_weighted_share - effective_target_share)
            / (raw_weighted_share - local_share_min)
        )

        adjusted_shares = [
            share - alpha * (share - local_share_min)
            for share in raw_shares
        ]

    # 3. Convert continuous quantities to integers.
    ideal_high_moisture = [q * share for q, share in zip(g_totals, adjusted_shares)]

    high_moisture = [int(math.floor(q)) for q in ideal_high_moisture]

    # Largest-remainder correction guarantees that the aggregate integer quantity exactly equals the desired target.
    # Nodes with largest lost fractional parts getting one additional unit of high-moisture waste until the target is reached.
    remainder = target_high_moisture - sum(high_moisture)

    fractional_parts = [
        ideal - integer
        for ideal, integer in zip(ideal_high_moisture, high_moisture)
    ]

    if remainder > 0:
        indices = sorted(
            range(len(g_totals)),
            key=lambda i: fractional_parts[i],
            reverse=True,
        )

        for i in indices[:remainder]:
            high_moisture[i] += 1

    Q_gw = [
        [
            high_moisture[g],
            g_totals[g] - high_moisture[g],
        ]
        for g in range(len(g_totals))
    ]

    return Q_gw
#endregion

#endregion

####################################################################################
############################### Data definition ####################################
####################################################################################

#region Instance generation
# stylized, empirically calibrated Chinese megacity systems
def generate_instance(
        params: InstanceParameters,
        instance_name: str,
        instance_size_class: str,
        instance_regime: str,
        seed: int = 7,
) -> InstanceData:
    rng = random.Random(seed)

    # -----------------------------
    # SETS
    # -----------------------------
    W_max = 2
    K_max = 3
    F_max = 1   # necessary to choose diferent coal types?
    H_max = 5

    # G = range(G_max)
    S = range(params.S_total)
    I = range(params.I_total)
    L = range(params.L_total)
    C = range(params.C_total)
    
    W = range(W_max)
    K = range(K_max)
    F = range(F_max)
    H = range(H_max)

    # "Anhui Conch Cement (cluster)", "Suzhou Dahua Marine", "Jiangsu Pengfei (Haian)", "Zhejiang Producer A", "Jiangsu Producer A","Anhui Producer A"
    cement_names = [f"Cement Plant {i+1}" for i in C]  # Placeholder names; replace with actual names if desired

    G_max = compute_grid_generation_count(
        city_size_x=params.city_size_x, 
        city_size_y=params.city_size_y, 
        cell_size=params.grid_cell_size
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
        city_size_x=params.city_size_x,
        city_size_y=params.city_size_y,
        grid_cell_size=params.grid_cell_size,
        incinerator_radius_min=params.incinerator_radius_min,
        incinerator_radius_max=params.incinerator_radius_max,
        incinerator_radius_center=params.incinerator_radius_center,
        landfill_radius_min=params.landfill_radius_min,
        landfill_radius_max=params.landfill_radius_max,
        landfill_radius_center=params.landfill_radius_center,
        cement_radius_min=params.cement_radius_min,
        cement_radius_max=params.cement_radius_max,
        cement_radius_center=params.cement_radius_center,
        incinerator_colocation_probability=params.incinerator_colocation_probability,
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
        clipping=params.clipping_distances,
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
        clipping=params.clipping_distances,
    )

    TD_si_avg = sum(TD_si[s][i] for s in S for i in I) / (params.S_total * params.I_total)
    
    TD_sl = build_transport_distance_matrix(
        origins=S_coords,
        destinations=L_coords,
        rng=rng,
        min_distance=30.0,
        max_distance=120.0,
        detour_min=1.10,
        detour_max=1.40,
        noise_share=0.04,
        clipping=params.clipping_distances,
    )

    TD_sc = build_transport_distance_matrix(
        origins=S_coords,
        destinations=C_coords,
        rng=rng,
        min_distance=60.0,
        max_distance=320.0,
        detour_min=1.10,
        detour_max=1.40,
        noise_share=0.04,
        clipping=params.clipping_distances,
    )

    # -----------------------------
    # EMISSIONS / COSTS
    # w=0: high moisture/chlorine, most organic and low fossil plastic content (worse)
    # w=1: RDF-like medium moisture/chlorine, higher fossil plastic content (better)
    # -----------------------------
    epsilon_truck = 0.0002          # tCO2e/t-km, refuse truck (heavy duty long haul truck 0.00006 tCO2e/t-km, but consider higher 
                                    # due to empty return trips, imperfect loading, stop-start operation, etc.)
    epsilon_land = [0.85, 0.60]
    # epsilon_inc = [0.55, 0.40]
    epsilon_inc = [0.20, 0.40]
    epsilon_kiln_w = [0.22, 0.68]
    epsilon_kiln_f = [2.25]   # epsilon_kiln_f = [2.25, 2.59]

    # -----------------------------
    # WASTE GENERATION (t/year):
    # Total MSW in a synthetic chinese Mega-city
    # E.g. Shanghai generates around 10,000,000–17,000,000 t/year in total
    # Split by type: 40-70% high moisture, rest medium moisture.
    # -----------------------------
    # total_target = rng.randint(6_000_000, 9_000_000)
    total_target = sample_total_waste_generation(
        rng=rng, 
        city_size_x=params.city_size_x, 
        city_size_y=params.city_size_y, 
        waste_density_t_per_km2_year=params.waste_gen_density
    )
    # split = [0.55, 0.45]      # fixed split
    # distribute by district (G) using a Dirichlet-like random split
    weights = [rng.random() for _ in G]
    weights_sum = sum(weights)
    weights = [w / weights_sum for w in weights]

    # Continuous node-level quantities.
    ideal_g_totals = [total_target * weights[g] for g in G]

    # Integer quantities using largest-remainder allocation.
    g_totals = [int(math.floor(quantity)) for quantity in ideal_g_totals ]

    remaining_waste = total_target - sum(g_totals)

    fractional_parts = [ideal - integer for ideal, integer in zip(ideal_g_totals, g_totals)]

    indices = sorted(
        range(len(g_totals)),
        key=lambda g: fractional_parts[g],
        reverse=True,
    )

    for g in indices[:remaining_waste]:
        g_totals[g] += 1

    # Allocate waste types while maintaining the prescribed aggregate high-moisture share.
    Q_gw = allocate_waste_composition(
        g_totals=g_totals,
        rng=rng,
        overall_high_moisture_share=params.high_moisture_share_total_network,
        local_share_min=params.local_high_moisture_bounds_node[0],
        local_share_max=params.local_high_moisture_bounds_node[1],
    )

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
        n_facilities=params.S_total,
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
        n_facilities=params.I_total,
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
        n_facilities=params.L_total,
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
    validate_system_capacity(
        Q_gen_total=Q_gen_total,
        Q_s=Q_s,
        Q_i=Q_i,
        Q_l=Q_l,
        Q_k=Q_k,
        C_total=params.C_total,
        kappa_land=params.kappa_land,
        kappa_coproc=params.kappa_coproc,
    )

    phi_wh = [[(h / (H_max - 1)) * params.phi_max_w[w] for h in H] for w in W]

    # U_w = [min(sum(Q_gw[g][w] for g in G), Q_k_max*len(C)) for w in W]  # Upper bound on waste flow of type w (can be tightened based on data)
    # A waste type w cannot flow trough network in an amount larger than: (i) total generated amount, (ii) total transfer-station capacity, (iii) total co-processing capacity
    U_w = [min(total_Q_gen_per_w[w], sum(Q_s), Q_k_max*len(C)) for w in W]  # Upper bound on waste flow of type w (can be tightened based on data)

    # Only 'budget_mun_availability' % of the maximum total potential waste flow to kilns can be subsidized supposing maximum subsidy levels, 
    # to create a more realistic budget constraint that requires trade-offs in subsidy allocation
    budget_municipality = params.budget_mun_availability * sum(params.phi_max_w[w] * U_w[w] for w in W)  # Set municipal budget based on maximum potential subsidy payout with some availability factor

    # -----------------------------
    # FOLLOWER: coal types, costs, kiln demands
    # -----------------------------
    # Coal mixtures: e.g., lower-grade and higher-grade thermal coal
    # price_f = [700.0, 850.0]
    beta_f = [23.0]      # GJ/t (two mixes)     # beta_f = [23.0, 27.0]

    # Kiln daily energy requirement
    # 2.500-5.000 t clinker / day (sometimes up to 10.000 t/day possible), 3 - 3.7 GJ/t clinker
    # daily energy requirement per plant: 7,500 - 18,500 GJ/day
    alpha_c_daily = [rng.triangular(7000, 18000, 15000) for _ in C]
    alpha_c = [alpha * 365 for alpha in alpha_c_daily]

    high_moisture_reduction_factor = 0.6            # high moisture waste effective usable energy per incoming tonne is 40% due to pre-processing (drying, shredding, filtering, RDF production, etc.)
    medium_moisture_reduction_factor = 0.3          # medium moisture waste effective usable energy per incoming tonne is 70% due to pre-processing (drying, shredding, filtering, RDF production, etc.)
    beta_w = [12.0*(1-high_moisture_reduction_factor), 16.0*(1-medium_moisture_reduction_factor)]   # GJ/t for waste types, adjusted by moisture content reduction in pre-processing

    # Investment CAPEX by option size (CNY) – extend to K=3
    c_invest_k = [90_000_000, 120_000_000, 300_000_000]

    # Only 'budget_cem_availability' % of the maximum total potential investment cost for co-processing is available
    budget_cem = params.budget_cem_availability * params.C_total * max(c_invest_k)  # bigger portfolio-level budget for 6 plants

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
        'F4': {c: (alpha_c[c]*params.kappa_coproc) + 1 for c in C},     # Maximmum energy content in co-processing
        'F5': Q_k_max+1,                        # Maximum co-processing quantity (not really needed, because x_ck_fixed is already fixed in the OC block, thus the maximal capacity is deterministic based on the fixed investment decision; keep it for fallback)
        'F6': {s: {w: float(min(sum(Q_gw[g][w] for g in G), Q_s[s])) for w in W} for s in S},  # Maximum waste flow from transfer station s to kiln c based on total generation and station capacity
        # since alpha_c is in GJ and beta_f is in GJ/t, a physically meaningful coal bound is closer to alpha_c[c] / beta_f[f] + 1.0
        'q_cf': {c: {f: alpha_c[c] / beta_f[f] + 1 for f in F} for c in C},   # Maximum quantity of coal processed at cement plant (based on maximum energy content needed)
        'q_scw': {s: {c: {w: float(min(Q_s[s], Q_k_max, total_Q_gen_per_w[w]))+1 for w in W} for c in C} for s in S},  # Maximum quantity of waste allocated from transfer station to cement plant
        # 'r_sw': {s: {w: float(min(Q_s[s], total_Q_gen_per_w[w]))+1 for w in W} for s in S},  # Maximum residual waste at transfer station after allocation, capcitated by individual capacities of transfer stations
    }

    lam_F3_bound = min(params.price_coal_f[f] / beta_f[f] for f in F)
    lam_F4_bound =  max(
        max(
            0.0,
            lam_F3_bound
            - (params.c_preproc_w[w]
                + params.c_truck * TD_sc[s][c]
                - params.c_penalty
                - params.phi_max_w[w]) 
            / beta_w[w]
        )
        for s in S
        for c in C
        for w in W
    )
    
    M_dual = {
        'lam_F3': lam_F3_bound + 1,     # Big-M for dual variable of constraint F3 (energy fulfillment constraint)
        'lam_F4': lam_F4_bound + 1,     # Big-M for dual variable of constraint F4 (maximum co-processing quantity)
        'lam_F5': params.bigM_duals_unrestricted,     # Big-M for dual variable of constraint F5 (co-process capacity limited by investment decision)
        'lam_F6': params.bigM_duals_unrestricted,     # Big-M for dual variable of constraint F6 (waste flow from transfer station to kiln limited by generation and station capacity)
        # derived from stationarity for q_cf: data.price_f[f] - lam_F3[c]*data.beta_f[f] - pi_q_cf[c,f] == 0 with lam_F3 >= 0 and beta_f >= 8, so price_f is a reasonable upper bound for pi_q_cf
        'pi_q_cf': max(params.price_coal_f)+1,    # Big-M for dual variable of constraint limiting quantity of coal processed at cement plant
        'pi_q_scw': params.bigM_duals_unrestricted,   # Big-M for dual variable of constraint limiting quantity of waste allocated from transfer station to cement plant
        # 'pi_r_sw': params.bigM_duals_unrestricted,    # Big-M for dual variable of constraint limiting residual waste at transfer station after allocation
    }

    return InstanceData(
        instance_name=instance_name,
        instance_size_class=instance_size_class,
        instance_regime=instance_regime,

        G_max=G_max, S_max=params.S_total, I_max=params.I_total, L_max=params.L_total, C_max=params.C_total,
        W_max=W_max, K_max=K_max, F_max=F_max, H_max=H_max,

        G=G, S=S, W=W, I=I, L=L, C=C, K=K, F=F, H=H,

        G_coords=G_coords, S_coords=S_coords, I_coords=I_coords, L_coords=L_coords, C_coords=C_coords,

        cement_names=cement_names,

        TD_gs=TD_gs, TD_sl=TD_sl, TD_si=TD_si, TD_si_avg=TD_si_avg, TD_sc=TD_sc,

        epsilon_truck=epsilon_truck,
        epsilon_land=epsilon_land, 
        epsilon_inc=epsilon_inc, 
        epsilon_kiln_w=epsilon_kiln_w, 
        epsilon_kiln_f=epsilon_kiln_f,

        c_truck=params.c_truck, 
        c_land=params.c_land, 
        c_inc=params.c_inc,
        
        Q_gw=Q_gw, 
        Q_gen_total=Q_gen_total,
        Q_s=Q_s, 
        Q_l=Q_l, 
        Q_i=Q_i,
        Q_k=Q_k, 
        Q_k_max=Q_k_max,
        
        kappa_land=params.kappa_land, 
        kappa_coproc=params.kappa_coproc,

        budget_municipality=budget_municipality,
        phi_max=params.phi_max_w,
        phi_wh=phi_wh,

        price_f=params.price_coal_f, 
        beta_f=beta_f,
        alpha_c=alpha_c, 
        beta_w=beta_w,

        c_invest_k=c_invest_k, 
        c_preproc_w=params.c_preproc_w,
        c_penalty=params.c_penalty, 
        budget_cem=budget_cem,
        fixcost_invest_k=fixcost_invest_k,

        M_primal=M_primal, 
        M_dual=M_dual,
        U_w=U_w
    )
#endregion

__all__ = ["InstanceData", "InstanceParameters", "generate_instance", "compute_grid_generation_count"]

if __name__ == "__main__":
    parameters = InstanceParameters(
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
        incinerator_colocation_probability=0.025,

        landfill_radius_min=40.0,
        landfill_radius_max=120.0,
        landfill_radius_center=75.0,

        cement_radius_min=80.0,
        cement_radius_max=320.0,
        cement_radius_center=200.0,

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
        bigM_duals_unrestricted=1e4,
    )

    instance = generate_instance(
        params=parameters,
        instance_name="synthetic_megacity_1",
        instance_size_class="medium",
        instance_regime="baseline",
        seed=7,
    )
    print(instance)