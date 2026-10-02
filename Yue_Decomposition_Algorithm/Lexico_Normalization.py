from dataclasses import dataclass
import logging
from typing import Literal

from gurobipy import GRB

from Instances.instance_generator import InstanceData
from .MP_BigM import MasterProblem


ObjectiveComponent = Literal["emission", "cost"]


@dataclass(frozen=True)
class PayoffPoint:
    """One lexicographically optimized anchor point."""
    emission: float
    cost: float


@dataclass(frozen=True)
class NormalizationBounds:
    """
    Normalization bounds obtained from the lexicographic payoff matrix:
        Emission_min = emission ideal value
        Emission_max = emission nadir value
        Cost_min     = cost ideal value
        Cost_max     = cost nadir value
    """
    Emission_min: float
    Emission_max: float
    Cost_min: float
    Cost_max: float


def _check_optimal(model, stage: str) -> None:
    """Raise an error if a normalization solve did not terminate optimally."""
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Normalization solve failed during '{stage}': "
            f"status={model.Status}, sol_count={model.SolCount}"
        )


def _lexicographic_tolerance(
    objective_value: float,
    *,
    abs_tol: float,
    rel_tol: float,
) -> float:
    """
    Compute numerical tolerance when fixing the primary objective before minimizing the secondary objective.
    """
    return max(
        abs_tol,
        rel_tol * max(1.0, abs(objective_value)),
    )


def _solve_lexicographic_anchor(
    instance: InstanceData,
    primary_objective: ObjectiveComponent,
    *,
    output_flag: int = 0,
    mip_gap: float = 1e-6,
    lex_abs_tol: float = 1e-6,
    lex_rel_tol: float = 1e-8,
) -> tuple[float, PayoffPoint]:
    """
    Compute one lexicographic payoff-matrix anchor.

    primary_objective == "emission":
        1. min E
        2. min C subject to E <= E* + tolerance

    primary_objective == "cost":
        1. min C
        2. min E subject to C <= C* + tolerance

    Returns
    -------
    primary_ideal:
        Optimal value of the primary objective from stage 1.

    payoff_point:
        Lexicographically refined anchor point after stage 2.
    """

    if primary_objective == "emission":
        secondary_objective = "cost"
    elif primary_objective == "cost":
        secondary_objective = "emission"
    else:
        raise ValueError(
            f"Unknown primary_objective: {primary_objective}. Must be 'emission' or 'cost'."
        )

    logging.info("\n" + "-" * 60)
    logging.info(
        f"Computing lexicographic anchor: min({primary_objective}) -> min({secondary_objective})",
        # "Computing lexicographic anchor: min(%s) -> min(%s)",
        # primary_objective,
        # secondary_objective,
    )
    logging.info("-" * 60)

    # ----------------------------------------------------------
    # Build normalization model
    # ----------------------------------------------------------

    mp = MasterProblem(instance)
    mp.build(
        name=f"Normalization_lex_{primary_objective}_{secondary_objective}",
        output_flag=output_flag,
    )

    model = mp.model

    # Raw leader-objective expressions are already constructed by build().
    E = mp.obj_total_env
    C = mp.obj_total_mon

    if E is None or C is None:
        raise RuntimeError(
            "Leader objective components were not initialized."
        )

    objective_expr = {
        "emission": E,
        "cost": C,
    }

    primary_expr = objective_expr[primary_objective]
    secondary_expr = objective_expr[secondary_objective]

    # Normalization anchors should be solved accurately.
    model.Params.MIPGap = mip_gap

    # Same numerical settings as used in the MP solve.
    model.Params.ScaleFlag = 2
    model.Params.Presolve = 2
    model.Params.NumericFocus = 1
    model.Params.IntegralityFocus = 1
    model.Params.FeasibilityTol = 1e-6

    # ==========================================================
    # Stage 1: optimize primary objective
    # ==========================================================

    model.setObjective(primary_expr, GRB.MINIMIZE)
    model.optimize()

    _check_optimal(
        model,
        stage=f"primary minimization of {primary_objective}",
    )

    primary_ideal = float(model.ObjVal)

    logging.info(
        "Primary optimum %s = %.10f",
        primary_objective,
        primary_ideal,
    )

    # ==========================================================
    # Stage 2: fix primary objective and optimize secondary
    # ==========================================================

    lex_tol = _lexicographic_tolerance(
        primary_ideal,
        abs_tol=lex_abs_tol,
        rel_tol=lex_rel_tol,
    )

    model.addConstr(
        primary_expr <= primary_ideal + lex_tol,
        name=f"LexFix_{primary_objective}",
    )

    model.setObjective(secondary_expr, GRB.MINIMIZE)
    model.update()
    model.optimize()

    _check_optimal(
        model,
        stage=(
            f"secondary minimization of {secondary_objective} "
            f"after fixing {primary_objective}"
        ),
    )

    payoff_point = PayoffPoint(
        emission=float(E.getValue()),
        cost=float(C.getValue()),
    )

    logging.info(
        "Lexicographic anchor min(%s) -> min(%s):",
        primary_objective,
        secondary_objective,
    )
    logging.info(
        "  Emissions = %.10f",
        payoff_point.emission,
    )
    logging.info(
        "  Costs     = %.10f",
        payoff_point.cost,
    )
    logging.info(
        "  Primary fixing tolerance = %.10e",
        lex_tol,
    )

    return primary_ideal, payoff_point


def determine_normalization_bounds(
    instance: InstanceData,
    *,
    output_flag: int = 0,
    mip_gap: float = 1e-6,
    lex_abs_tol: float = 1e-6,
    lex_rel_tol: float = 1e-8,
) -> NormalizationBounds:
    """
    Determine normalization bounds using a lexicographic payoff matrix.

    Payoff matrix:

                            Emissions       Costs
        lex min(E, C)       E_ideal         C_nadir
        lex min(C, E)       E_nadir         C_ideal

    Thus:
        Emission_min = E_ideal
        Emission_max = E_nadir
        Cost_min     = C_ideal
        Cost_max     = C_nadir
    """

    logging.info("\n" + "#" * 60)
    logging.info(
        "Computing normalization bounds from lexicographic payoff matrix..."
    )
    logging.info("#" * 60)

    # ----------------------------------------------------------
    # Anchor 1: lexicographically minimize (Emissions, Costs)
    # ----------------------------------------------------------
    Emission_ideal, emission_anchor = _solve_lexicographic_anchor(
        instance,
        primary_objective="emission",
        output_flag=output_flag,
        mip_gap=mip_gap,
        lex_abs_tol=lex_abs_tol,
        lex_rel_tol=lex_rel_tol,
    )

    # At this anchor:
    #   emissions = ideal emissions
    #   costs     = cost nadir
    Cost_nadir = emission_anchor.cost

    # ----------------------------------------------------------
    # Anchor 2: lexicographically minimize (Costs, Emissions)
    # ----------------------------------------------------------
    Cost_ideal, cost_anchor = _solve_lexicographic_anchor(
        instance,
        primary_objective="cost",
        output_flag=output_flag,
        mip_gap=mip_gap,
        lex_abs_tol=lex_abs_tol,
        lex_rel_tol=lex_rel_tol,
    )

    # At this anchor:
    #   costs     = ideal costs
    #   emissions = emission nadir
    Emission_nadir = cost_anchor.emission

    # ----------------------------------------------------------
    # Store bounds
    # ----------------------------------------------------------

    bounds = NormalizationBounds(
        Emission_min=Emission_ideal,
        Emission_max=Emission_nadir,
        Cost_min=Cost_ideal,
        Cost_max=Cost_nadir,
    )

    # Keep existing InstanceData interface unchanged.
    instance.Emission_min = bounds.Emission_min
    instance.Emission_max = bounds.Emission_max
    instance.Cost_min = bounds.Cost_min
    instance.Cost_max = bounds.Cost_max

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    logging.info("\n" + "=" * 60)
    logging.info("Lexicographic payoff matrix")
    logging.info("=" * 60)
    logging.info(
        "%-22s %18s %18s",
        "",
        "Emissions",
        "Costs",
    )
    logging.info(
        "%-22s %18.6f %18.6f",
        "lex min(E, C)",
        bounds.Emission_min,
        bounds.Cost_max,
    )
    logging.info(
        "%-22s %18.6f %18.6f",
        "lex min(C, E)",
        bounds.Emission_max,
        bounds.Cost_min,
    )

    logging.info("-" * 60)
    logging.info("Normalization ranges:")
    logging.info(
        "  Emissions: [%.6f, %.6f], range = %.6f",
        bounds.Emission_min,
        bounds.Emission_max,
        bounds.Emission_max - bounds.Emission_min,
    )
    logging.info(
        "  Costs:     [%.6f, %.6f], range = %.6f",
        bounds.Cost_min,
        bounds.Cost_max,
        bounds.Cost_max - bounds.Cost_min,
    )
    logging.info("=" * 60 + "\n")

    return bounds