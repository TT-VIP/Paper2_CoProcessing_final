from dataclasses import dataclass
import logging
from typing import Literal

import gurobipy as gp
from gurobipy import GRB

from Instances.instance_generator import InstanceData
from .MP_BigM import MasterProblem


ObjectiveComponent = Literal["emission", "cost"]
SenseType = Literal["min", "max"]


@dataclass(frozen=True)
class NormalizationBounds:
    """Container for min-max normalization bounds of the leader objective."""
    Emission_min: float
    Emission_max: float
    Cost_min: float
    Cost_max: float


def _solve_single_objective_bound(
    instance: InstanceData,
    objective_component: ObjectiveComponent,
    sense: SenseType,
    *,
    output_flag: int = 0,
    # time_limit: float = GRB.INFINITY,
    # mip_gap: float = 1e-6,
) -> float:
    """
    Build the relaxed master problem and solve one single-objective normalization model.

    objective_type:
        "emission" -> optimize total raw leader emissions E(x)
        "cost"     -> optimize total raw leader costs C(x)

    sense:
        "min" -> minimize selected objective
        "max" -> maximize selected objective
    """

    # Temporarily deactivate normalization bounds to avoid using normalized objective coefficients in this auxiliary solve, 
    # which is only used to determine the normalization bounds themselves. Restore the original bounds after this solve.
    # old_bounds = (
    #     instance.Emission_min,
    #     instance.Emission_max,
    #     instance.Cost_min,
    #     instance.Cost_max,
    # )

    # instance.Emission_min = None
    # instance.Emission_max = None
    # instance.Cost_min = None
    # instance.Cost_max = None

    mp = MasterProblem(instance)
    mp.build(name=f"Normalization_{objective_component}_{sense}", output_flag=output_flag) # within build() the objective is set depending on the current instance bounds, but we will overwrite it below for this auxiliary solve

    # if objective_component == "emission" and sense == "max":
    #     for c in instance.C:
    #         for f in instance.F:
    #             mp.q_cf0[c, f].UB = instance.M_primal["q_cf"][c][f]
    #     mp.model.update()
    mp.model.addConstrs(
        (gp.quicksum(mp.q_cf0[c, f] * instance.beta_f[f] for f in instance.F)
        + gp.quicksum(mp.q_scw0[s, c, w] * instance.beta_w[w] for s in instance.S for w in instance.W )
        == instance.alpha_c[c] for c in instance.C
    ),name="Norm_energyFulfillmentEquality",
    )

    mp.model.update()

    E, C = mp._build_leader_objective_components()

    if objective_component == "emission":
        objective_expr = E
    elif objective_component == "cost":
        objective_expr = C
    else:
        raise ValueError(f"Unknown objective_component: {objective_component}.\n"
                         f"Must be 'emission' or 'cost'.")

    # Overwrite and set objective
    gurobi_sense = GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE
    mp.model.setObjective(objective_expr, gurobi_sense)
    mp.model.update()

    # mp.model.Params.TimeLimit = time_limit
    # mp.model.Params.MIPGap = mip_gap
    
    # Control scaling of linear optimization problems (optional), see https://link.springer.com/article/10.1007/s10589-011-9420-4
    # mp.model.Params.ScaleFlag = 2

    mp.model.Params.Threads = 4  # Use 4 threads for parallel processing to speed up the solve time
    mp.model.optimize()

    if mp.model.status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Normalization bound solve failed: "
            f"component={objective_component}, sense={sense}, "
            f"status={mp.model.status}, sol_count={mp.model.SolCount}"
        )

    mp_sol = mp.extract_solution()

    # if mp.model.SolCount == 0:
    #     # instance.Emission_min, instance.Emission_max, instance.Cost_min, instance.Cost_max = old_bounds
    #     raise RuntimeError(
    #         f"Could not determine normalization bound for "
    #         f"{objective_component=}, {sense=}."
    #     )
    
    logging.info(f"Solution status for {objective_component=}, {sense=}: {mp.model.Status}")
    logging.info("\nObjective breakdown MP:\n")
    if objective_component == "emission":
        indices_to_report = {1, 2, 3, 4}
    elif objective_component == "cost":
        indices_to_report = {6, 7, 8, 9}

    for index, (component, value) in enumerate(mp_sol.objective_components.items(), start=1):
        if index in indices_to_report:
            logging.info(f"{component:<30} {float(value):>14.6f}")
            if index in (4, 9):  # Add extra spacing after transport and treatment costs for readability
                logging.info("")
    
    value = float(mp.model.ObjVal)
    return value

    # Restore previous bounds after this auxiliary solve.
    # instance.Emission_min, instance.Emission_max, instance.Cost_min, instance.Cost_max = old_bounds


def determine_normalization_bounds(
    instance: InstanceData,
    *,
    output_flag: int = 0,
) -> NormalizationBounds:
    """
    Compute and store the four min-max normalization bounds for one instance.

    The bounds are computed on the relaxed master problem:
        Emission_min = min E(x)
        Emission_max = max E(x)
        Cost_min     = min C(x)
        Cost_max     = max C(x)
    """

    logging.info("\n" + "#" * 60)
    logging.info("Computing normalization bounds on relaxed MP...")
    logging.info("#" * 60)

    Emission_min = _solve_single_objective_bound(
        instance, "emission", "min",
        output_flag=output_flag,
    )

    Emission_max = _solve_single_objective_bound(
        instance, "emission", "max",
        output_flag=output_flag,
    )

    Cost_min = _solve_single_objective_bound(
        instance, "cost", "min",
        output_flag=output_flag,
    )

    Cost_max = _solve_single_objective_bound(
        instance, "cost", "max",
        output_flag=output_flag,
    )

    bounds = NormalizationBounds(
        Emission_min=Emission_min,
        Emission_max=Emission_max,
        Cost_min=Cost_min,
        Cost_max=Cost_max,
    )

    instance.Emission_min = bounds.Emission_min
    instance.Emission_max = bounds.Emission_max
    instance.Cost_min = bounds.Cost_min
    instance.Cost_max = bounds.Cost_max

    logging.info("-" * 60)
    logging.info("Normalization bounds computed:")
    logging.info(f"  Emission_min = {bounds.Emission_min:.6f}")
    logging.info(f"  Emission_max = {bounds.Emission_max:.6f}")
    logging.info(f"  Cost_min     = {bounds.Cost_min:.6f}")
    logging.info(f"  Cost_max     = {bounds.Cost_max:.6f}")
    logging.info("-" * 60 + "\n")

    return bounds