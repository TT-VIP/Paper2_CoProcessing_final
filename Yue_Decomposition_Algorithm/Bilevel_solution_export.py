from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from Instances.instance_generator import InstanceData
from .Decomposition_Algorithm import DecompositionSolution


# ============================================================================
# Variable index definitions
# ============================================================================

LEADER_INDEXED_VARIABLES: dict[str, tuple[str, ...]] = {
    "q_gsw": ("g", "s", "w"),
    "q_slw": ("s", "l", "w"),
    "q_siw": ("s", "i", "w"),
    "d_siw": ("s", "i", "w"),
    "z_wh": ("w", "h"),
    "y_wh": ("w", "h"),
}

LEADER_SCALAR_VARIABLES: tuple[str, ...] = (
    "mu_land",
    "mu_inc",
    "mu_kiln",
)

FOLLOWER_INDEXED_VARIABLES: dict[str, tuple[str, ...]] = {
    "x_ck": ("c", "k"),
    "q_cf": ("c", "f"),
    "r_sw": ("s", "w"),
    "q_scw": ("s", "c", "w"),
}


# ============================================================================
# General JSON helpers
# ============================================================================

def _finite_float(value: Any) -> float | None:
    """
    Convert a numerical value to a JSON-safe Python float.

    Non-finite values such as +/- infinity or NaN are represented as None,
    which becomes null in JSON.
    """
    if value is None:
        return None

    result = float(value)

    if not math.isfinite(result):
        return None

    return result


def _json_index_value(value: Any) -> Any:
    """
    Convert an index value to a native JSON-compatible scalar.

    This mainly protects against NumPy integer/string scalar types.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def _make_json_safe(value: Any) -> Any:
    """
    Recursively convert common Python/NumPy values into JSON-compatible types.

    Useful for algorithm configuration dictionaries.
    """
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return _finite_float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return _finite_float(value)

    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]

    return value


# ============================================================================
# Variable serialization
# ============================================================================

def _normalize_index(index: Any) -> tuple:
    """
    Convert a scalar or tuple index into a tuple.

    Examples
    --------
    (0, 1, 2) -> (0, 1, 2)
    3         -> (3,)
    """
    if isinstance(index, tuple):
        return index

    return (index,)


def _serialize_indexed_variable(
    values: Mapping,
    index_names: Sequence[str],
    *,
    include_zeros: bool = True,
    zero_tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    """
    Convert an indexed solution dictionary into a list of indexed records.

    Example
    -------
    {
        (0, 2, 1): 50.0
    }

    becomes

    [
        {
            "s": 0,
            "c": 2,
            "w": 1,
            "value": 50.0
        }
    ]
    """
    records: list[dict[str, Any]] = []

    for index, raw_value in sorted(values.items()):
        index_tuple = _normalize_index(index)

        if len(index_tuple) != len(index_names):
            raise ValueError(
                "Variable index dimension does not match index specification: "
                f"index={index_tuple}, "
                f"expected {len(index_names)} dimensions "
                f"({tuple(index_names)})."
            )

        value = float(raw_value)

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite decision-variable value encountered: index={index_tuple}, value={value}."
            )

        if not include_zeros and abs(value) <= zero_tolerance:
            continue

        record = {
            index_name: _json_index_value(index_value)
            for index_name, index_value
            in zip(index_names, index_tuple)
        }

        # Deliberately add value last for readability.
        record["value"] = value

        records.append(record)

    return records


def _serialize_indexed_variables(
    solution_object: Any,
    variable_specification: Mapping[str, Sequence[str]],
    *,
    include_zeros: bool,
    zero_tolerance: float,
) -> dict[str, list[dict[str, Any]]]:
    """
    Serialize multiple indexed variables from a solution object.
    """
    return {
        variable_name: _serialize_indexed_variable(
            getattr(solution_object, variable_name),
            index_names,
            include_zeros=include_zeros,
            zero_tolerance=zero_tolerance,
        )
        for variable_name, index_names
        in variable_specification.items()
    }


def _serialize_scalar_variables(
    solution_object: Any,
    variable_names: Sequence[str],
) -> dict[str, float]:
    """
    Serialize scalar decision variables.
    """
    return {
        variable_name: float(
            getattr(solution_object, variable_name)
        )
        for variable_name in variable_names
    }


# ============================================================================
# Objective-component serialization
# ============================================================================

def _serialize_objective_components(
    components: Mapping[str, Any] | None,
) -> dict[str, float]:
    """
    Convert objective-component values to plain Python floats.
    """
    if not components:
        return {}

    return {
        str(component): float(value)
        for component, value in components.items()
    }


# ============================================================================
# Iteration-history serialization
# ============================================================================

def _serialize_iteration_history(
    decomp_sol: DecompositionSolution,
) -> list[dict[str, Any]]:
    """
    Convert IterationRecord objects into JSON-compatible dictionaries.
    """
    history: list[dict[str, Any]] = []

    for record in decomp_sol.iteration_history:
        history.append(
            {
                "iteration": int(record.iteration),

                "mp_obj": _finite_float(record.mp_obj),
                "mp_bound": _finite_float(record.mp_bound),
                "mp_mip_gap": _finite_float(record.mp_mip_gap),

                "sp1_obj": _finite_float(record.sp1_obj),
                "sp1_obj_original": _finite_float(record.sp1_obj_original),

                "sp2_obj": _finite_float(record.sp2_obj),
                "sp2_feasible": record.sp2_feasible,

                "lb": _finite_float(record.lower_bound),
                "ub": _finite_float(record.upper_bound),
                "gap_abs": _finite_float(record.gap_abs),
                "gap_rel": _finite_float(record.gap_rel),

                "mp_runtime": _finite_float(record.mp_runtime),
                "sp1_runtime": _finite_float(record.sp1_runtime),
                "sp2_runtime": _finite_float(record.sp2_runtime),
                "total_iteration_time": _finite_float(record.total_iteration_time),
                "total_time": _finite_float(record.total_time),

                "mp_mode": record.mp_mode,

                "lb_updated": bool(record.lb_updated),
                "ub_updated": bool(record.ub_updated),

                "new_pattern_added": bool(record.new_pattern_added),
                "duplicate_pattern": bool(record.duplicate_pattern),
            }
        )

    return history


# ============================================================================
# Complete solution dictionary
# ============================================================================

def build_solution_dictionary(
    *,
    decomp_sol: DecompositionSolution,
    instance: InstanceData,
    method_tag: str,
    algorithm_config: Mapping[str, Any],
    include_zeros: bool = True,
    zero_tolerance: float = 1e-8,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Build the complete machine-readable decomposition solution.

    Parameters
    ----------
    decomp_sol:
        Final decomposition result returned by run_yue_decomposition().

    instance:
        Instance data corresponding to the run.

    method_tag:
        Short identifier of the algorithmic variant, e.g. "BigM1e5".

    algorithm_config:
        Dictionary containing solver/decomposition parameters.

    include_zeros:
        If True, store all indexed variable values.
        If False, omit values satisfying abs(value) <= zero_tolerance.

    zero_tolerance:
        Numerical threshold used only when include_zeros=False.

    timestamp:
        Optional ISO-format timestamp. If omitted, the current local
        timestamp is used.
    """
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative.")

    if timestamp is None:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    solution: dict[str, Any] = {
        "schema_version": 1,

        "metadata": {
            "instance_name": instance.instance_name,
            "size_class": instance.instance_size_class,
            "structural_regime": instance.instance_regime,
            "method_tag": method_tag,
            "timestamp": timestamp,

            "network_dimensions": {
                "G": int(instance.G_max),
                "S": int(instance.S_max),
                "I": int(instance.I_max),
                "L": int(instance.L_max),
                "C": int(instance.C_max),
            },
        },

        "variable_storage": {
            "representation": "indexed_records",
            "include_zeros": bool(include_zeros),
            "zero_tolerance": float(zero_tolerance),
            "missing_records_are_zero": not include_zeros,
        },

        "algorithm_config": _make_json_safe(
            dict(algorithm_config)
        ),

        "result": {
            "status": decomp_sol.status.name,
            "termination_reason": decomp_sol.termination_reason,

            "iterations": int(decomp_sol.iterations),
            "completed_iterations": len(decomp_sol.iteration_history),

            "max_iterations": int(decomp_sol.max_iterations),

            "iteration_best_solution": decomp_sol.iteration_best_solution,

            "total_solution_time": _finite_float(decomp_sol.total_solution_time),

            "lower_bound": _finite_float(decomp_sol.lower_bound),

            "upper_bound": _finite_float(decomp_sol.upper_bound),

            "final_gap_proven": _finite_float(decomp_sol.final_gap_proven),

            "final_gap_incumbents": _finite_float(decomp_sol.final_gap_incumbents),
        },

        "iteration_history": _serialize_iteration_history(decomp_sol),

        "best_solution": None,
    }

    # ------------------------------------------------------------------
    # No bilevel-feasible solution available
    # ------------------------------------------------------------------
    if (
        decomp_sol.best_bilevel_mp_sol is None
        or decomp_sol.best_bilevel_sp2_sol is None
    ):
        return solution

    mp_sol = decomp_sol.best_bilevel_mp_sol
    sp2_sol = decomp_sol.best_bilevel_sp2_sol

    # ------------------------------------------------------------------
    # Leader variables
    # ------------------------------------------------------------------
    leader_variables = _serialize_indexed_variables(
        mp_sol,
        LEADER_INDEXED_VARIABLES,
        include_zeros=include_zeros,
        zero_tolerance=zero_tolerance,
    )

    leader_variables.update(
        _serialize_scalar_variables(
            mp_sol,
            LEADER_SCALAR_VARIABLES,
        )
    )

    # ------------------------------------------------------------------
    # Follower variables
    # ------------------------------------------------------------------
    follower_variables = _serialize_indexed_variables(
        sp2_sol,
        FOLLOWER_INDEXED_VARIABLES,
        include_zeros=include_zeros,
        zero_tolerance=zero_tolerance,
    )

    # ------------------------------------------------------------------
    # Best solution
    # ------------------------------------------------------------------
    solution["best_solution"] = {
        "leader": {
            # Bilevel-feasible leader objective / current upper bound.
            "objective":
                _finite_float(sp2_sol.sp2_obj),

            "objective_components":
                _serialize_objective_components(sp2_sol.leader_objective_components),

            "variables": 
                leader_variables,
        },

        "follower": {
            "objective_reduced":
                _finite_float(sp2_sol.follower_obj_reduced),

            "objective_original":
                _finite_float(sp2_sol.follower_obj_original),

            "objective_components":
                _serialize_objective_components(sp2_sol.follower_objective_components),

            "variables": 
                follower_variables,
        },

        "associated_master_solution": {
            "objective":
                _finite_float(mp_sol.mp_obj),

            "bound":
                _finite_float(mp_sol.mp_bound),

            "objective_components":
                _serialize_objective_components(mp_sol.objective_components),
        },
    }

    return solution


# ============================================================================
# JSON writer
# ============================================================================

def write_solution_json(
    solution: Mapping[str, Any],
    output_path: Path | str,
) -> None:
    """
    Write a decomposition solution dictionary to a JSON file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            solution,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )