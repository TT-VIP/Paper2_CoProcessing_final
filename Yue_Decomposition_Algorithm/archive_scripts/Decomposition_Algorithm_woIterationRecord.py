import numpy as np
import math
import logging
from datetime import datetime
from pathlib import Path
import gurobipy as gp
import time
from enum import Enum, auto         # define a set of named constant values for decomposition status
from dataclasses import dataclass
from typing import Optional

from Instances.instance_generator import InstanceData
from ..MP_BigM import MasterProblem as BigMMasterProblem, MasterSolution
from ..MP_SOS1 import MasterProblem as SOS1MasterProblem
from ..SP1 import SubProblem1, SubProblem1Solution
from ..SP2 import SubProblem2, SubProblem2Solution


class DecompositionStatus(Enum):
    OPTIMAL_PROVEN = auto()
    SUBOPTIMAL_MPSP2_INCUMBENTS_MATCH = auto()
    FEASIBLE_SUBOPTIMAL = auto()
    NO_BILEVEL_FEASIBLE_SOLUTION = auto()
    MP_INFEASIBLE_OR_NO_SOLUTION = auto()
    NUMERICAL_BOUND_INCONSISTENCY = auto()

@dataclass
class DecompositionSolution:
    status: DecompositionStatus             # Overall status of the decomposition algorithm at termination (e.g., optimality proven, feasible but not proven optimal, no feasible solution found, etc.)

    xi: float                               # Convergence threshold for leader objective improvement (used for termination)
    max_iterations: int                     # Maximum number of iterations allowed for the decomposition algorithm (used for termination)
    iterations: int = 0                     # Actual number of iterations performed
    iteration_best_solution: Optional[int] = None  # Iteration number at which the best solution (lowest feasible UB) was found, if applicable
    total_solution_time: float = 0.0        # Total time taken for the entire decomposition algorithm (from start to termination)

    lower_bound: float = -np.inf            # Best lower bound on the leader's objective value found at termination of decomposition (from MP)
    upper_bound: float = np.inf             # Best upper bound on the leader's objective value found at termination of decomposition (from SP2)
    final_gap_proven: float = np.inf        # Final true optimality gap (UB - LB) at termination, if applicable
    final_gap_incumbents: float = np.inf    # Final gap based on incumbent solutions (SP2 obj - MP obj) at termination, if applicable

    equality_tol: float = 1e-3

    best_bilevel_mp_sol: Optional["MasterSolution"] = None
    best_bilevel_sp2_sol: Optional["SubProblem2Solution"] = None

    termination_reason: Optional[str] = None


def build_decomposition_solution(
    *,
    iterations: int,
    max_iterations: int,
    iteration_best_solution: Optional[int],
    total_solution_time: float,
    LB: float,
    UB: float,
    Xi: float,
    best_bilevel_mp_sol,
    best_bilevel_sp2_sol,
    termination_reason: str,
    equality_tol: float = 1e-3,
):
    best_bilevel_mp_obj = None if best_bilevel_mp_sol is None else float(best_bilevel_mp_sol.mp_obj)
    best_bilevel_sp2_obj = None if best_bilevel_sp2_sol is None else float(best_bilevel_sp2_sol.sp2_obj)

    final_gap_proven = None
    if math.isfinite(LB) and math.isfinite(UB):
        final_gap_proven = UB - LB
    
    final_gap_incumbents = None
    if best_bilevel_mp_obj is not None and best_bilevel_sp2_obj is not None:
        final_gap_incumbents = abs(best_bilevel_sp2_obj - best_bilevel_mp_obj)

    if math.isfinite(LB) and math.isfinite(UB) and (final_gap_proven < -1e-8):
        status = DecompositionStatus.NUMERICAL_BOUND_INCONSISTENCY
    elif math.isfinite(LB) and math.isfinite(UB) and (final_gap_proven <= Xi):
        status = DecompositionStatus.OPTIMAL_PROVEN
    elif (
        best_bilevel_mp_obj is not None
        and best_bilevel_sp2_obj is not None
        and final_gap_incumbents <= equality_tol
    ):
        status = DecompositionStatus.SUBOPTIMAL_MPSP2_INCUMBENTS_MATCH
    elif best_bilevel_sp2_sol is not None:
        status = DecompositionStatus.FEASIBLE_SUBOPTIMAL
    elif termination_reason == "MP solution run returned no feasible solution":
        status = DecompositionStatus.MP_INFEASIBLE_OR_NO_SOLUTION
    else:
        status = DecompositionStatus.NO_BILEVEL_FEASIBLE_SOLUTION

    return DecompositionSolution(
        status=status,
        xi=Xi,
        max_iterations=max_iterations,
        iterations=iterations,
        iteration_best_solution=iteration_best_solution,
        total_solution_time=total_solution_time,
        lower_bound=LB,
        upper_bound=UB,
        final_gap_proven=final_gap_proven,
        final_gap_incumbents=final_gap_incumbents,
        equality_tol=equality_tol,
        best_bilevel_mp_sol=best_bilevel_mp_sol,
        best_bilevel_sp2_sol=best_bilevel_sp2_sol,
        termination_reason=termination_reason
    )


#region Big-M Check 
# Check Big-M binding in OC blocks
def log_bigM_binding(mp: BigMMasterProblem, data: InstanceData, *, tol_ratio: float = 1e-3) -> None:
    """
    Logs if any big-M caps appear binding in any OC block.
    
    Reduced follower model:
    - r_sw, pi_r_sw, and bin_r_sw are eliminated.
    - lam_F6[s,w] and bin_F6[s,w] correspond to the station-availability slack

          A_sw - sum_c q_scw[s,c,w] >= 0.

    Parameters
    mp:         Master problem object after solving.
    data:       Instance data.
    tol_ratio:  flag as binding if value >= (1 - tol_ratio) * M
    """
    if not hasattr(mp, "kkt_oc_blocks") or not mp.kkt_oc_blocks:
        return

    def is_one(x: float) -> bool:
        return x >= 0.5  # binary; tolerate numerics

    def near_cap(val: float, M: float) -> bool:
        if M is None or M <= 0 or val is None:
            return False
        return val >= (1.0 - tol_ratio) * M

    # Pull leader z-values once (used in F9 b-terms)
    z_val = {(w, h): float(mp.z_wh[w, h].X) for w in data.W for h in data.H}

    total_dual_hits = 0
    total_primal_hits = 0

    for l, oc in mp.kkt_oc_blocks.items():
        dual_hits = []
        primal_hits = []

        # -------- helper: check lam <= M_dual * bin
        def check_dual_cap(name: str, lam_var, bin_var, M_key: str) -> None:
            nonlocal total_dual_hits

            if M_key not in data.M_dual:
                logging.info(f"[BigM] Missing M_dual[{M_key}] for checking dual cap in {name}. Skipping this check.")
                return
            M = float(data.M_dual[M_key])
            lam = float(lam_var.X)
            b = float(bin_var.X)

            if is_one(b) and near_cap(lam, M):
                dual_hits.append(f"{name}: dual={lam:.6g} hits M={M:.6g}")
                total_dual_hits += 1

        def _get_primal_M(data: InstanceData, M_key: str, *idx) -> float:
            M = data.M_primal[M_key]
            for i in idx:
                M = M[i]
            return float(M)

        # -------- helper: check b_expr <= M_primal*(1-bin)
        def check_primal_cap(name: str, b_expr: float, bin_var, M_key: str, *idx) -> None:
            nonlocal total_primal_hits

            if M_key not in data.M_primal:
                logging.info(f"[BigM] Missing M_primal[{M_key}] for checking primal cap in {name}. Skipping this check.")
                return
            # M = float(data.M_primal[M_key])
            M = _get_primal_M(data, M_key, *idx)
            b = float(bin_var.X)

            if (not is_one(b)) and near_cap(b_expr, M):
                primal_hits.append(f"{name}: primal={b_expr:.6g} hits M={M:.6g}")
                total_primal_hits += 1

        # === F3 slack
        # (b = sum_f q_cf*beta_f + sum_sw q_scw*beta_w - alpha_c) >= 0
        for c in data.C:
            bF3 = sum(float(oc.q_cf[c, f].X) * data.beta_f[f] for f in data.F) + sum(float(oc.q_scw[s, c, w].X) * data.beta_w[w] for s in data.S for w in data.W) - data.alpha_c[c]
            check_dual_cap(f"OC{l}.F3[c={c}]", oc.lam_F3[c], oc.bin_F3[c], "lam_F3")
            check_primal_cap(f"OC{l}.F3[c={c}]", bF3, oc.bin_F3[c], "F3")

        # === F4 slack
        # (b = kappa_coproc*alpha - sum_sw q_scw*beta_w) >= 0
        for c in data.C:
            bF4 = data.kappa_coproc * data.alpha_c[c] - sum(float(oc.q_scw[s, c, w].X) * data.beta_w[w] for s in data.S for w in data.W)
            check_dual_cap(f"OC{l}.F4[c={c}]", oc.lam_F4[c], oc.bin_F4[c], "lam_F4")
            check_primal_cap(f"OC{l}.F4[c={c}]", bF4, oc.bin_F4[c], "F4", c)

        # === F5 slack
        # (b = sum_k x_ck_fixed*Q_k - sum_sw q_scw) >= 0
        for c in data.C:
            cap = sum(oc.x_ck_fixed[(c, k)] * data.Q_k[k] for k in data.K)
            bF5 = cap - sum(float(oc.q_scw[s, c, w].X) for s in data.S for w in data.W)
            check_dual_cap(f"OC{l}.F5[c={c}]", oc.lam_F5[c], oc.bin_F5[c], "lam_F5")
            check_primal_cap(f"OC{l}.F5[c={c}]", bF5, oc.bin_F5[c], "F5")

            # In the MP formulation, F5 uses cap_c directly as the primal big-M.
            # Therefore, compare against cap_c rather than a global M if desired.
            b = float(oc.bin_F5[c].X)
            if (not is_one(b)) and cap > 0 and near_cap(bF5, float(cap)):
                primal_hits.append(
                    f"OC{l}.F5[c={c}]: primal_slack={bF5:.6g} hits cap_c={float(cap):.6g}"
                )
                total_primal_hits += 1

        # === F6 slack (reduced station-availability)
        # A_sw - sum_c q_scw >= 0
        for s in data.S:
            for w in data.W:
                A_sw = (sum(float(mp.q_gsw[g, s, w].X) for g in data.G)
                    - sum(float(mp.q_slw[s, ll, w].X) for ll in data.L)
                    - sum(float(mp.q_siw[s, i, w].X) for i in data.I))
                slack_F6 = (A_sw - sum(float(oc.q_scw[s, c, w].X) for c in data.C))
                check_dual_cap(f"OC{l}.F6[s={s},w={w}]",oc.lam_F6[s, w],oc.bin_F6[s, w],"lam_F6")
                check_primal_cap(f"OC{l}.F6[s={s},w={w}]",slack_F6,oc.bin_F6[s, w],"F6",s,w)

        # === Bound complementarity examples: pi_q_cw <= M*pi * bin, and q_cw <= M*q * (1-bin)
        # pi_q_cf * q_cf = 0
        for c in data.C:
            for f in data.F:
                check_dual_cap(f"OC{l}.pi_q_cf[{c},{f}]", oc.pi_q_cf[c, f], oc.bin_q_cf[c, f], "pi_q_cf")
                q = float(oc.q_cf[c, f].X)
                check_primal_cap(f"OC{l}.q_cf[{c},{f}]", q, oc.bin_q_cf[c, f], "q_cf", c, f)

        # pi_q_scw * q_scw = 0
        for s in data.S:
            for c in data.C:
                for w in data.W:
                    check_dual_cap(f"OC{l}.pi_q_scw[{s},{c},{w}]", oc.pi_q_scw[s, c, w], oc.bin_q_scw[s, c, w], "pi_q_scw")
                    q = float(oc.q_scw[s, c, w].X)
                    check_primal_cap(f"OC{l}.q_scw[{s},{c},{w}]", q, oc.bin_q_scw[s, c, w], "q_scw", s, c, w)


        if dual_hits or primal_hits:
            logging.info(f"[BigM] OC block l={l}: dual_hits={len(dual_hits)}, primal_hits={len(primal_hits)}")
            for s in dual_hits:
                logging.info(f"  - {s}")
            for s in primal_hits:
                logging.info(f"  - {s}")
    if total_dual_hits == 0 and total_primal_hits == 0:
        logging.info("[BigM] No big-M caps appear binding at the chosen tolerance.")
#endregion

#region SOS1 Check
# Check Duality Cut Weakness in OC blocks
def log_sos1_primal_dual_residuals(mp, data: InstanceData, *, tol: float = 1e-5) -> None:
    """
    Diagnostic for SOS1 KKT blocks.

    Compares:
        theta_tilde
        exact dual expression using current A_sw
        safe dual expression using U_A_sw

    Main quantities:
        exact_residual = theta_tilde - dual_exact
        safe_residual  = theta_tilde - dual_safe
        conservatism   = dual_exact - dual_safe
                       = sum_{s,w} (U_A_sw - A_sw) * lambda_F6[s,w] >= 0

    Interpretation:
        - exact_residual near 0: KKT block is internally coherent.
        - safe_residual large but exact_residual near 0: Kleinert cut is weak due to U_A_sw.
        - exact_residual significantly negative: likely modelling/sign/numerical issue.
        - exact_residual significantly positive: KKT/SOS1 not tight at incumbent or degeneracy/numerics.
    """
    if not hasattr(mp, "kkt_oc_blocks") or not mp.kkt_oc_blocks:
        logging.info("[SOS1-PD] No KKT-OC blocks available.")
        return

    def qgen_w(w: int) -> float:
        return float(sum(data.Q_gw[g][w] for g in data.G))

    def structural_U_A(s: int, w: int) -> float:
        return float(min(qgen_w(w), data.Q_s[s]))

    logging.info("\n[SOS1-PD] Exact primal-dual residual diagnostics")

    worst_abs_exact_residual = 0.0
    worst_block = None

    for ell, oc in mp.kkt_oc_blocks.items():

        # ------------------------------------------------------------
        # Pattern-specific installed capacity
        # ------------------------------------------------------------
        cap_c = {
            c: float(sum(int(round(oc.x_ck_fixed[(c, k)])) * data.Q_k[k] for k in data.K))
            for c in data.C
        }

        # ------------------------------------------------------------
        # Current leader-induced availability A_sw at MP incumbent
        # ------------------------------------------------------------
        A_sw = {}

        for s in data.S:
            for w in data.W:
                A_sw[(s, w)] = (
                    sum(float(mp.q_gsw[g, s, w].X) for g in data.G)
                    - sum(float(mp.q_slw[s, l, w].X) for l in data.L)
                    - sum(float(mp.q_siw[s, i, w].X) for i in data.I)
                )

        u_sw_f6 = getattr(oc, "U_sw_F6_l", None)
        U_A_sw = dict(u_sw_f6) if u_sw_f6 is not None else {}
        if not U_A_sw:
            # Fallback: use structural upper bound if U_A_sw not stored in OC block
            U_A_sw = {(s, w): structural_U_A(s, w) for s in data.S for w in data.W}

        # ------------------------------------------------------------
        # theta_tilde: reduced primal objective in the KKT block
        # ------------------------------------------------------------
        theta_tilde = (
            sum(
                float(oc.q_cf[c, f].X) * data.price_f[f]
                for c in data.C for f in data.F
            )
            + sum(
                float(oc.q_scw[s, c, w].X)
                * (
                    data.c_preproc_w[w]
                    + data.c_truck * data.TD_sc[s][c]
                    - data.c_penalty
                )
                for s in data.S for c in data.C for w in data.W
            )
            - sum(
                data.phi_wh[w][h] * float(oc.y_wh_KKT[w, h].X)
                for w in data.W for h in data.H
            )
        )

        # ------------------------------------------------------------
        # Exact dual value using current A_sw
        # ------------------------------------------------------------
        dual_exact = (
            sum(data.alpha_c[c] * float(oc.lam_F3[c].X) for c in data.C)
            - sum(
                data.kappa_coproc * data.alpha_c[c] * float(oc.lam_F4[c].X)
                for c in data.C
            )
            - sum(cap_c[c] * float(oc.lam_F5[c].X) for c in data.C)
            - sum(
                A_sw[(s, w)] * float(oc.lam_F6[s, w].X)
                for s in data.S for w in data.W
            )
        )

        # ------------------------------------------------------------
        # Safe dual value used in the implemented Kleinert inequality
        # ------------------------------------------------------------
        dual_safe = (
            sum(data.alpha_c[c] * float(oc.lam_F3[c].X) for c in data.C)
            - sum(
                data.kappa_coproc * data.alpha_c[c] * float(oc.lam_F4[c].X)
                for c in data.C
            )
            - sum(cap_c[c] * float(oc.lam_F5[c].X) for c in data.C)
            - sum(
                U_A_sw[(s, w)] * float(oc.lam_F6[s, w].X)
                for s in data.S for w in data.W
            )
        )

        exact_residual = theta_tilde - dual_exact
        safe_residual = theta_tilde - dual_safe
        conservatism = dual_exact - dual_safe

        worst_abs_exact_residual = max(worst_abs_exact_residual, abs(exact_residual))
        if worst_abs_exact_residual == abs(exact_residual):
            worst_block = ell

        # ------------------------------------------------------------
        # Optional detail: which F6 rows cause conservatism?
        # ------------------------------------------------------------
        f6_terms = []
        for s in data.S:
            for w in data.W:
                lam = float(oc.lam_F6[s, w].X)
                if abs(lam) > tol:
                    gap_A = U_A_sw[(s, w)] - A_sw[(s, w)]
                    term = gap_A * lam
                    f6_terms.append((abs(term), s, w, A_sw[(s, w)], U_A_sw[(s, w)], lam, term))

        f6_terms.sort(reverse=True)

        logging.info(
            f"[SOS1-PD] OC{ell}: "
            f"theta={theta_tilde:.6f}, "
            f"dual_exact={dual_exact:.6f}, "
            f"dual_safe={dual_safe:.6f}, "
            f"exact_residual={exact_residual:.6e}, "
            f"safe_residual={safe_residual:.6e}, "
            f"conservatism={conservatism:.6e}"
        )

        if exact_residual < -tol:
            logging.warning(
                f"[SOS1-PD] WARNING OC{ell}: exact residual is negative "
                f"({exact_residual:.6e}). Check signs, stationarity, or numerical tolerances."
            )

        if f6_terms:
            logging.info(f"[SOS1-PD] OC{ell}: largest F6 conservatism terms:")
            for _, s, w, A_val, U_val, lam, term in f6_terms[:5]:
                logging.info(
                    f"  (s={s}, w={w}): "
                    f"A={A_val:.6f}, U_A={U_val:.6f}, "
                    f"lambda_F6={lam:.6f}, "
                    f"(U_A-A)*lambda={term:.6e}"
                )
        else:
            logging.info(f"[SOS1-PD] OC{ell}: no positive lambda_F6 rows above tol={tol:g}.")

    logging.info(
        f"[SOS1-PD] Worst absolute exact residual: "
        f"{worst_abs_exact_residual:.6e} in OC{worst_block}"
    )

#region Logging Helper
# Helper functions for pattern keys and solution logging
# convert x_ck_fixed dict to a sorted tuple for consistent pattern keys in logging and cut management
def pattern_key(x_ck_fixed: dict) -> tuple:
    # sort to be deterministic
    return tuple(sorted((c, k, int(round(v))) for (c, k), v in x_ck_fixed.items()))

def format_pattern_dict(x_ck: dict) -> str:
    items = ", ".join(f"({c}, {k}): {int(round(v))}" for (c, k), v in sorted(x_ck.items()))
    return items

def log_nonzero_gurobi_vars(model: gp.Model, model_name: str, tol: float = 1e-4, var_names_to_log: list = None) -> None:
    """Log all nonzero variable values of a solved Gurobi model."""
    # logging.info(f"Objective value: {model.ObjVal:.4f}")

    logging.info(f"\nNonzero variables in {model_name} (|x| > {tol}):")
    count = 0
    for v in model.getVars():
        val = v.X
        if abs(val) > tol:
            # If filter list provided, only log if variable name starts with one of the filters
            if var_names_to_log is not None:
                if not any(v.VarName.startswith(name) for name in var_names_to_log):
                    continue
            logging.info(f"  {v.VarName} = {val:.10g}")
            count += 1
    logging.info(f"Total nonzero vars in {model_name}: {count}")

def log_objective_components(title: str, components: dict[str, float] | None) -> None:
    """Pretty-print objective components if available."""
    if not components:
        logging.info(f"\n{title}: no objective component breakdown available.")
        return

    logging.info(f"\n{title}:\n")
    for component, value in components.items():
        logging.info(f"{component:<45} {float(value):>14.6f}")


def log_sp1_solution(sp1_sol: SubProblem1Solution) -> None:
    """Log SP1 solution details under the reduced follower formulation."""
    logging.info(f"Subproblem 1 reduced follower objective: {sp1_sol.sp1_obj:.5f}")

    if hasattr(sp1_sol, "sp1_obj_original"):
        logging.info(
            f"Subproblem 1 reconstructed original follower objective: "
            f"{sp1_sol.sp1_obj_original:.5f}"
        )

    logging.info(f"Binary combination in SP1: x_ck = {sp1_sol.x_ck}")

    if getattr(sp1_sol, "objective_components", None) is not None:
        log_objective_components(
            "Objective breakdown SP1 follower",
            sp1_sol.objective_components,
        )


def log_sp2_solution(sp2_sol: SubProblem2Solution) -> None:
    """Log SP2 solution details under the reduced follower formulation."""
    if not sp2_sol.feasible:
        logging.info("Subproblem 2 infeasible.")
        return

    logging.info(f"Subproblem 2 leader objective: {sp2_sol.sp2_obj:.5f}")

    if getattr(sp2_sol, "follower_obj_reduced", None) is not None:
        logging.info(
            f"Subproblem 2 reduced follower objective: "
            f"{sp2_sol.follower_obj_reduced:.5f}"
        )

    if getattr(sp2_sol, "follower_obj_original", None) is not None:
        logging.info(
            f"Subproblem 2 reconstructed original follower objective: "
            f"{sp2_sol.follower_obj_original:.5f}"
        )

    logging.info(f"Binary combination in SP2: x_ck = {sp2_sol.x_ck}")

    if getattr(sp2_sol, "objective_components", None) is not None:
        log_objective_components(
            "Objective breakdown SP2 follower",
            sp2_sol.follower_objective_components,
        )

def log_duplicate_pattern_diagnostic(
        mp,
        sp1_sol: SubProblem1Solution,
        data: InstanceData,
        *,
        follower_objective_scale: float = 1_000_000,
) -> None:
    """
    Diagnostic for a duplicate SP1 follower investment pattern.

    Compares, at the current MP incumbent:
        1) SP1 global optimal reduced follower objective
        2) MP dummy reduced follower objective
        3) reduced follower objective of the matching KKT-OC block

    If the SP1 pattern has already been enumerated, the matching KKT block
    should represent the optimal continuous follower reaction conditional
    on exactly this investment pattern.

    Expected relationship, up to numerical tolerances:
        f_SP1 <= f_dummy
        f_KKT_matching ≈ f_SP1

    and, through the Yue optimality cut,
        f_dummy <= f_KKT_matching + tolerance.
    """

    if mp.model is None or mp.model.SolCount == 0:
        logging.warning("Duplicate-pattern diagnostic skipped: no MP incumbent available.")
        return

    # ----------------------------------------------------------
    # 1) Find the OC block corresponding to the SP1 pattern
    # ----------------------------------------------------------
    sp1_key = pattern_key(sp1_sol.x_ck)

    matching_blocks = [(ell, oc) for ell, oc in mp.kkt_oc_blocks.items() if pattern_key(oc.x_ck_fixed) == sp1_key]

    if not matching_blocks:
        logging.warning("Duplicate-pattern diagnostic failed: SP1 pattern is marked as previously generated, but no matching KKT-OC block was found.")
        return

    if len(matching_blocks) > 1:
        logging.warning(f"Duplicate-pattern diagnostic: found {len(matching_blocks)} KKT-OC blocks for the same pattern. Using the first one."
        )

    ell, oc = matching_blocks[0]

    # ----------------------------------------------------------
    # 2) SP1 globally optimal reduced follower objective
    # ----------------------------------------------------------
    f_sp1 = float(sp1_sol.sp1_obj)

    # ----------------------------------------------------------
    # 3) MP dummy reduced follower objective
    #
    # Must reproduce exactly the LHS of the Yue optimality cut
    # in MP_BigM.py.
    # ----------------------------------------------------------
    f_dummy = (
        # Coal cost
        sum(float(mp.q_cf0[c, f].X) * data.price_f[f]
            for c in data.C
            for f in data.F
        )

        # Investment cost
        + sum(data.fixcost_invest_k[k] * float(mp.x_ck0[c, k].X)
            for c in data.C
            for k in data.K
        )

        # Pre-processing cost
        + sum(float(mp.q_scw0[s, c, w].X) * data.c_preproc_w[w]
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Transport cost to cement kilns
        + sum(float(mp.q_scw0[s, c, w].X) * data.c_truck * data.TD_sc[s][c]
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Reduced residual-waste penalty contribution
        - sum(float(mp.q_scw0[s, c, w].X) * data.c_penalty
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Subsidy revenue
        - sum(float(mp.y_wh[w, h].X) * data.phi_wh[w][h]
            for w in data.W
            for h in data.H
        )
    )

    # ----------------------------------------------------------
    # 4) Matching KKT-block reduced follower objective
    #
    # Must reproduce exactly the RHS of the Yue optimality cut.
    # ----------------------------------------------------------
    f_kkt = (
        # Coal cost
        sum(float(oc.q_cf[c, f].X) * data.price_f[f]
            for c in data.C
            for f in data.F
        )

        # Investment cost for fixed pattern
        + sum(data.fixcost_invest_k[k] * oc.x_ck_fixed[(c, k)]
            for c in data.C
            for k in data.K
        )

        # Pre-processing cost
        + sum(float(oc.q_scw[s, c, w].X) * data.c_preproc_w[w]
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Transport cost
        + sum(float(oc.q_scw[s, c, w].X)
            * data.c_truck
            * data.TD_sc[s][c]
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Reduced residual-waste penalty contribution
        - sum(float(oc.q_scw[s, c, w].X) * data.c_penalty
            for s in data.S
            for c in data.C
            for w in data.W
        )

        # Subsidy revenue
        - sum(float(oc.y_wh_KKT[w, h].X) * data.phi_wh[w][h]
            for w in data.W
            for h in data.H
        )
    )

    # ----------------------------------------------------------
    # 5) Gaps
    # ----------------------------------------------------------
    gap_dummy_sp1 = f_dummy - f_sp1
    gap_kkt_sp1 = f_kkt - f_sp1
    gap_dummy_kkt = f_dummy - f_kkt

    # Current dummy investment pattern is useful context:
    dummy_pattern = {
        (c, k): int(round(mp.x_ck0[c, k].X))
        for c in data.C
        for k in data.K
    }

    # ----------------------------------------------------------
    # 6) Compact diagnostic logging
    # ----------------------------------------------------------
    logging.info("\n" + "=" * 70)
    logging.info("DUPLICATE PATTERN DIAGNOSTIC")
    logging.info("=" * 70)

    logging.info(f"SP1 pattern:                     {format_pattern_dict(sp1_sol.x_ck)}")
    logging.info(f"MP dummy pattern x_ck0:          {format_pattern_dict(dummy_pattern)}")
    logging.info(f"Matching OC block:               OC{ell}")

    logging.info("")
    logging.info(f"SP1 optimal LL objective:        {f_sp1:,.6f}")
    logging.info(f"MP dummy LL objective:           {f_dummy:,.6f}")
    logging.info(f"Matching KKT LL objective:       {f_kkt:,.6f}")

    logging.info("")
    logging.info(f"Gap dummy - SP1:                 {gap_dummy_sp1:,.6f}")
    logging.info(f"Gap KKT   - SP1:                 {gap_kkt_sp1:,.6f}")
    logging.info(f"Gap dummy - KKT:                 {gap_dummy_kkt:,.6f}")

    logging.info("")
    logging.info(
        f"Scaled gaps (/ {follower_objective_scale:g}): "
        f"dummy-SP1={gap_dummy_sp1 / follower_objective_scale:.6e}, "
        f"KKT-SP1={gap_kkt_sp1 / follower_objective_scale:.6e}, "
        f"dummy-KKT={gap_dummy_kkt / follower_objective_scale:.6e}"
    )

    logging.info("=" * 70)
#endregion

#region Decomposition Alg
def run_yue_decomposition(
        Verbose: bool = True,
        mp_normal_time_limit: float = 180,
        mp_polish_time_limit: float = 600,
        lb_stall_trigger: int = 2,
        # lb_progrerss_tol: float | None = None,
        sp1_max_time: float = 60,
        sp2_max_time: float = 60,
        mip_gap: float = 1e-4,
        Xi: float = 1e-1,
        max_iterations: int = 5,
        instance: InstanceData = None,
        weight_env: float = 0.5,
        weight_mon: float = 0.5,
        total_time_limit: float = 3630.0,
        shutdown_buffer: float = 30.0,
        objective_scale: float = 1.0,
        bigM_duals_unrestricted: float = 1e4,
        sos1_cuts: bool = False,
        primal_dual_strenghtening: bool = True,
        bound_cutoff: bool = True,
        cutoff_bound_tolerance: float = 1e-5,
        solution_dir: Optional[Path] = None,
) -> None:

    # Load instance data
    # shanghai_data = make_shanghai_instance_effective()
    if instance is None:
        raise RuntimeError("Instance data must be provided to run the Decomposition Algorithm.")
    instance_data = instance

    instance_data.weight_env = weight_env
    instance_data.weight_mon = weight_mon

    # Update algorithm-specific generic Big-M bounds for unrestricted dual variables.
    # Analytically derived bounds such as lam_F3 and pi_q_cf are intentionally left unchanged.
    if not sos1_cuts:
        if not math.isfinite(bigM_duals_unrestricted) or bigM_duals_unrestricted <= 0:
            raise ValueError(
                f"Big-M values of unrestricted dual variables ('bigM_duals_unrestricted') must be a positive finite value, got {bigM_duals_unrestricted}."
            )

        generic_dual_bigM_keys = (
            "lam_F4",
            "lam_F5",
            "lam_F6",
            "pi_q_scw",
            "pi_r_sw",
        )

        missing_keys = [
            key
            for key in generic_dual_bigM_keys
            if key not in instance_data.M_dual
        ]

        if missing_keys:
            raise KeyError(
                "Missing expected generic dual Big-M entries in instance data: "
                f"{missing_keys}"
            )

        for key in generic_dual_bigM_keys:
            instance_data.M_dual[key] = float(bigM_duals_unrestricted)

    # Starting Configuration
    LB = -np.inf
    UB = np.inf
    iteration = 0
    oc_blocks_added = 0
    duplicate_oc_blocks_skipped = 0

    # start timer for overall algorithm
    start_total = time.perf_counter()
    # internal wall-clock budget for solver calls
    def elapsed() -> float:
        return time.perf_counter() - start_total

    def remaining() -> float:
        return max(0.0, total_time_limit - elapsed())

    def time_left_for_solve() -> float:
        return max(1.0, remaining() - shutdown_buffer)

    # Initialize Master Problem - L=empty set is implicit: MP starts without any OC blocks
    MasterProblemClass = SOS1MasterProblem if sos1_cuts else BigMMasterProblem
    mp = MasterProblemClass(instance_data)
    mp.build(output_flag=1, objective_scale=objective_scale)  # scale objective to help with numerical issues and big-M binding detection in early iterations

    if sos1_cuts:
        logging.info("\nComputing LP-based availability bounds U_A_sw for SOS1 primal-dual cuts...")
        mp.compute_availability_bounds_lp(time_limit_per_lp=10.0, output_flag=0)

    best_bilevel_mp_sol = None
    best_bilevel_sp2_sol = None
    termination_reason = None
    iteration_best_solution = None

    # Track if we've printed quality for each model after first solve
    # mp_quality_printed = False
    sp1_statistics_printed = False
    sp2_statistics_printed = False

    generated_patterns_kkt_blocks = set()   # book-keeping: to track which patterns have had KKT OC blocks added, to avoid duplicates
    generated_patterns = []                 # list of dictionaries of all patterns in the order the cuts were added

    lb_stall_count = 0
    lb_progress_tol = max(0.1 * Xi, 1e-6)  # Minimum meaningful LB improvement threshold
    
    previous_iteration_duplicate = False
    previous_iteration_added_pattern = False
    previous_iteration_meaningful_lb_improvement = True

    # Decomposition Algorithm with KKT OC Cuts
    while iteration < max_iterations and (UB - LB > Xi) and remaining() > shutdown_buffer:
        iteration += 1
        terminate = False
        lb_updated = False
        ub_updated = False

        duplicate_pattern_this_iteration = False
        new_pattern_this_iteration = False
        meaningful_lb_improvement_this_iteration = False

        starttime_iteration = time.perf_counter()

        if Verbose:
            logging.info("\n" + "="*150)
            logging.info(f"Iteration {iteration}")
            logging.info(f"Current bounds: LB = {LB:.5f}, UB = {UB:.5f}, Gap = {(UB - LB):.5f}")
            logging.info("="*150)

        # if iteration <= 8:
        #     base_mp_limit = 300
        # elif iteration <= 10:
        #     base_mp_limit = 300
        # else:
        #     base_mp_limit = 600

        # mp_time_limit = min(base_mp_limit, time_left_for_solve())
        
        # if mp_time_limit <= 5.5:
        #     termination_reason = "Global time limit reached (before next MP solve, time left <= 5 seconds)"
        #     break

        # # Solve Master Problem
        # if iteration % 5 == 0:      # besser: Wenn LB in letzten beiden Iterationen nicht verbessert wurde, dann MIPFocus=3 setzen, um die Bound zu verbessern
        #     mp.model.Params.MIPFocus = 3  # Focus on best objective bound if bound is moving very slowly (or not at all)
        #     # mp.model.Params.ScaleFlag = 2  # Enable aggressive scaling to help with numerical issues and potentially improve bounds
        #     solver_time = min(mp_normal_time_limit, time_left_for_solve())

        #     logging.info("\n" + "="*70)
        #     logging.info(f"Master Problem Statistics Report (Iteration {iteration}):")
        #     logging.info("="*70)
        #     mp.model.printStats()
        #     logging.info("="*70 + "\n")

        #     mp.solve(time_limit=solver_time, mip_gap=mip_gap)  # Longer time limit for MP every 5 iterations to improve LB
        # else:
        #     mp.model.Params.MIPFocus = 0  # Default focus - balance between finding good solutions and proving optimality
        #     # mp.model.Params.MIPFocus = 2  # solver is having no trouble finding good quality solutions, and wish to focus more attention on proving optimality
        #     # mp.model.Params.ScaleFlag = 2   # Already default in MP.py
        #     mp.model.Params.Seed = 1

        #     logging.info("\n" + "="*70)
        #     logging.info(f"Master Problem Statistics Report (Iteration {iteration}):")
        #     logging.info("="*70)
        #     mp.model.printStats()
        #     logging.info("="*70 + "\n")

        #     mp.solve(time_limit=mp_time_limit, mip_gap=mip_gap)

        # ============================================================
        # Adaptive Master Problem solution strategy
        # ============================================================
        polish_reasons = []

        # If a genuinely new KKT block was added in the previous iteration, give the changed MP a normal balanced solve.
        fresh_structural_information = previous_iteration_added_pattern

        if not fresh_structural_information:
            # A duplicate means that the previous iteration produced no new KKT information. 
            # If the LB also did not improve meaningfully, invest additional effort into the existing MP.
            if (previous_iteration_duplicate and not previous_iteration_meaningful_lb_improvement):
                polish_reasons.append("duplicate follower pattern without LB progress")

            # Independent trigger: persistent LB stagnation.
            if lb_stall_count >= lb_stall_trigger:
                polish_reasons.append(f"LB unchanged for {lb_stall_count} consecutive iterations")


        bound_polishing = len(polish_reasons) > 0


        if bound_polishing:
            requested_mp_time = mp_polish_time_limit
            mip_focus = 3
            mp_mode = "BOUND POLISHING"
        else:
            requested_mp_time = mp_normal_time_limit
            mip_focus = 0
            mp_mode = "EXPLORATION"


        # ========================================================================================
        # Reserve enough global time to evaluate the resulting MP incumbent in SP1 and SP2.
        # ========================================================================================
        reserved_after_mp = sp1_max_time + sp2_max_time + shutdown_buffer

        available_mp_time = max(0.0, remaining() - reserved_after_mp)

        mp_time_limit = min(requested_mp_time, available_mp_time)

        if mp_time_limit <= 5.0:
            termination_reason = ("Global time limit reached before next MP solve (insufficient time for MP + subsequent SP1/SP2 evaluation).")
            break

        # Reproducible Gurobi configuration
        mp.model.Params.MIPFocus = mip_focus
        # mp.model.Params.Seed = 1

        logging.info("\n" + "=" * 70)
        logging.info(f"Master Problem Strategy - Iteration {iteration}")
        logging.info("=" * 70)
        logging.info(f"Mode:                    {mp_mode}")
        logging.info(f"MP time limit:           {mp_time_limit:.1f} s")
        logging.info(f"MIPFocus:                {mip_focus}")
        logging.info(f"LB stagnation counter:   {lb_stall_count}")

        if polish_reasons:
            logging.info("Polishing trigger:       " + "; ".join(polish_reasons))
        elif fresh_structural_information:
            logging.info("Normal mode reason:      new KKT-OC block added in previous iteration")
        else:
            logging.info("Normal mode reason:      no stagnation trigger")

        logging.info("=" * 70)

        logging.info("\n" + "=" * 70)
        logging.info(f"Master Problem Statistics Report (Iteration {iteration}):")
        logging.info("=" * 70)
        mp.model.printStats()
        logging.info("=" * 70 + "\n")

        mp.solve(time_limit=mp_time_limit,mip_gap=mip_gap)

        if mp.model.SolCount == 0:
            logging.info("No solution found for Master Problem. Terminating.")
            termination_reason = "MP solution run returned no feasible solution"
            break

        # Print MP solution quality after first solve (happens after first OC block is added in iteration 2)
        # if not mp_quality_printed and mp.model.SolCount > 0 and iteration >= 2:
        if mp.model.SolCount > 0:
            logging.info("\n" + "="*70)
            logging.info(f"Master Problem Solution Quality (Iteration {iteration}):")
            logging.info("="*70)
            mp.model.printQuality()
            logging.info("="*70 + "\n")
            # mp_quality_printed = True
        
        # ============================================================
        # LB update
        # ============================================================
        prev_LB = LB
        try:
            new_LB = mp.model.ObjBound  # Update LB with the best bound from MP
        except Exception:
            new_LB = -np.inf  # Fallback to -infinity if bound is not available (incumbent not feasible because it can overestimate the follower's objective)
        LB = max(LB, new_LB)  # Ensure LB does not decrease
        lb_updated = LB > prev_LB

        # ============================================================
        # Meaningful LB progress for adaptive MP strategy
        # ============================================================
        if math.isfinite(prev_LB):
            lb_improvement = LB - prev_LB
            meaningful_lb_improvement_this_iteration = (lb_improvement > lb_progress_tol)
        else:
            # First finite lower bound is always meaningful progress
            lb_improvement = np.inf
            meaningful_lb_improvement_this_iteration = math.isfinite(LB)

        if meaningful_lb_improvement_this_iteration:
            lb_stall_count = 0
        else:
            lb_stall_count += 1
        
        # solution logging
        logging.info(f"\nBest Master Problem Solution: Objective = {mp.model.ObjVal:.5f}, Bound = {mp.model.ObjBound:.5f}, Gap = {mp.model.MIPGap*100:.2f}%")
        if LB > prev_LB:
            logging.info(f"New LB found. LB updated from {prev_LB:.5f} to {LB:.5f}")
            logging.info(f"LB improvement this iteration: {lb_improvement:.8f} (meaningful threshold = {lb_progress_tol:.8f})")
        else:
            logging.info(f"LB stagnated: LB = {LB:.5f}")
            logging.info(f"Consecutive LB-stagnation iterations: {lb_stall_count}")

        mp_sol = mp.extract_solution()
        # Log the objective components for the MP solution
        logging.info("\nObjective breakdown MP:\n")
        logging.info(f"Weights: Environment ={instance_data.weight_env:.2f}, Monetary={instance_data.weight_mon:.2f}\n")
        for index, (component, value) in enumerate(mp_sol.objective_components.items(), start=1):
                logging.info(f"{component:<30} {float(value):>14.6f}")
                if index in (5,10):  # Add extra spacing after transport and treatment costs for readability
                    logging.info("")

        
        if not sos1_cuts:  # Big-M only relevant if not using SOS1 cuts
            logging.info(f"\nCheck big-M bindings in MP solution:")
            log_bigM_binding(mp, instance_data)        # Log any big-M bindings in the current MP solution
        else:
            log_sos1_primal_dual_residuals(mp, instance_data)  # Log primal-dual residual diagnostics for SOS1 KKT blocks
        
        if remaining() <= shutdown_buffer:
            termination_reason = "Global time limit reached (after MP solve and before SP solves staerted)"
            break

        sp1_time_limit = min(sp1_max_time, time_left_for_solve())
        # Solve Subproblem 1 at leader solution (Follower Optimality)
        sp1 = SubProblem1(instance_data)
        sp1.build(mp_sol, name=f"Subproblem 1 - Iteration {iteration}", output_flag=1)

        # ============================================================
        # Print SP1 statistics and solution quality
        # ============================================================
        if not sp1_statistics_printed:
            logging.info("\n" + "="*70)
            logging.info("Subproblem 1 Statistics:")
            logging.info("="*70)
            sp1.model.printStats()
            logging.info("="*70 + "\n")
            sp1_statistics_printed = True       # SP1 remains the same across iterations

        sp1.solve(time_limit=sp1_time_limit)

        # Print SP1 quality after first solve
        # if not sp1_statistics_printed and sp1.model.SolCount > 0:
        if sp1.model.SolCount > 0:
            logging.info("\n" + "="*70)
            logging.info(f"Subproblem 1 Solution Quality (Iteration {iteration}):")
            logging.info("="*70)
            sp1.model.printQuality()
            logging.info("="*70 + "\n")
            # sp1_statistics_printed = True

        sp1_sol = sp1.extract_solution()
        log_sp1_solution(sp1_sol)  # Log SP1 solution details, including objective breakdown if available
        # logging.info(f"Subproblem 1 Solution: {sp1_sol.sp1_obj:.5f}")
        # logging.info(f'Binary combination in SP1: x_ck = {sp1_sol.x_ck}')

        sp2_time_limit = min(sp2_max_time, time_left_for_solve())
        # Solve Subproblem 2 (Bilevel Feasibility) at leader solution and SP1 follower solution
        sp2 = SubProblem2(instance_data)
        sp2.build(mp_sol, sp1_sol, name=f"Subproblem 2 - Iteration {iteration}", output_flag=1, objective_scale=objective_scale)  # scale objective to help with numerical issues and big-M binding detection in early iterations

        # ============================================================
        # Print SP2 statistics and solution quality
        # ============================================================
        if not sp2_statistics_printed:
            logging.info("\n" + "="*70)
            logging.info("Subproblem 2 Statistics:")
            logging.info("="*70)
            sp2.model.printStats()
            logging.info("="*70 + "\n")
            sp2_statistics_printed = True       # SP2 remains the same across iterations

        # sp2.solve(time_limit=solver_time_limit)
        sp2.solve(time_limit=sp2_time_limit)

        # Print SP2 quality after first solve
        # if not sp2_statistics_printed and sp2.model.SolCount > 0:
        if sp2.model.SolCount > 0:
            logging.info("\n" + "="*70)
            logging.info(f"Subproblem 2 Solution Quality (Iteration {iteration}):")
            logging.info("="*70)
            sp2.model.printQuality()
            logging.info("="*70 + "\n")
            # sp2_statistics_printed = True

        sp2_sol = sp2.extract_solution()
        log_sp2_solution(sp2_sol)  # Log SP2 solution details, including objective breakdown if available

        # ============================================================
        # Update Upper Bound and Add KKT Optimality Cut if SP2 is feasible
        # ============================================================
        if sp2_sol.feasible:
            # Update upper bound and best solutions if better
            if float(sp2_sol.sp2_obj) < UB:
                prev_UB = UB
                UB = float(sp2_sol.sp2_obj)
                ub_updated = True
                best_bilevel_mp_sol = mp_sol
                best_bilevel_sp2_sol = sp2_sol
                iteration_best_solution = iteration
                logging.info(f"\nSubproblem 2 feasible. New UB found. UB updated from {prev_UB:.5f} to {UB:.5f}")
                # logging.info(f'Binary combination in SP2: x_ck = {sp2_sol.x_ck}')
            else:
                logging.info(f"Subproblem 2 feasible but no improvement. Upper Bound remains unchanged: UB = {UB:.5f}")
                # logging.info(f'Binary combination in SP2: x_ck = {sp2_sol.x_ck}')
            
            # Check convergence or finish before adding cut
            bound_consistency_tol = 1e-8
            bilevel_gap = UB - LB

            if bilevel_gap < -bound_consistency_tol:
                termination_reason = (f"ERROR: Numerically inconsistent bounds detected:"
                                      f" UB = {UB:.12f} < LB = {LB:.12f}, difference = {bilevel_gap:.3e}"
                                      f" Check solver tolerances and numerical stability.")
                logging.error(termination_reason)
                terminate = True
            
            elif bilevel_gap <= Xi:
                logging.info(f"Convergence achieved: UB - LB <= Xi: {UB - LB:.5f} <= {Xi}")
                logging.info("Terminating decomposition algorithm.")
                termination_reason = "Convergence achieved based on consistent bounds and Gap tolerance (UB - LB <= Xi)"
                terminate = True
            elif iteration == max_iterations:
                logging.info(f"Maximum iterations reached: {iteration}. Terminating decomposition algorithm.")
                termination_reason = "Maximum iterations reached without convergence."
                terminate = True
            
            # check if x_ck KKT pattern has already had a KKT OC block added; if so, skip adding another to force diversification in future iterations
            if not terminate:
                key = pattern_key(sp2_sol.x_ck)
                if key in generated_patterns_kkt_blocks:
                    duplicate_pattern_this_iteration = True
                    logging.info("ATTENTION: Duplicate x_ck pattern from SP2 encountered. OC block will NOT be duplicated because of no improvement. Solution run will be continued in the next iteration.")
                    duplicate_oc_blocks_skipped += 1
                    # logging.info("Duplicate x_ck pattern from SP2 encountered. Skipping OC block and forcing diversification.")
                    # mp._add_no_good_cut(sp2_sol.x_ck)  # Add no-good cut to forbid this exact x_ck pattern in future iterations
                else:
                    new_pattern_this_iteration = True
                    generated_patterns_kkt_blocks.add(key)
                    generated_patterns.append(sp2_sol.x_ck)  # Store the pattern for logging and analysis
                    # Add KKT Optimality Cut to MP based on SP2 solution
                    logging.info("Adding KKT-OC block based on x_ck of SP2 solution to cut off current leader solution.")
                    if sos1_cuts == True:
                        mp._add_kkt_oc_block_sos1(sp2_sol.x_ck, primal_dual_streghtening=primal_dual_strenghtening)
                    else:
                        mp._add_kkt_oc_block_bigM(sp2_sol.x_ck)
                    oc_blocks_added += 1
        
        else:
            if iteration == max_iterations:
                logging.info(f"Maximum iterations reached: {iteration}. Terminating decomposition algorithm.")
                termination_reason = "Maximum iterations reached without convergence."
                terminate = True

            else:
                sp2.model.computeIIS()
                iis_dir = Path(solution_dir) / "SP2_IIS" if solution_dir is not None else Path("SP2_IIS")
                iis_dir.mkdir(parents=True, exist_ok=True)
                ilp_path = iis_dir / f"SP2_infeasible_{iteration}.ilp"
                # ilp_name = f"SP2_infeasible_{iteration}.ilp"
                # ilp_path = str(Path(solution_dir) / ilp_name) if solution_dir is not None else ilp_name
                sp2.model.write(str(ilp_path))
                logging.info(f"Subproblem 2 is infeasible -> IIS written to {ilp_path}. Upper bound remains unchanged.")
                key = pattern_key(sp1_sol.x_ck)
                if key in generated_patterns_kkt_blocks:
                    duplicate_pattern_this_iteration = True
                    logging.info("ATTENTION: Duplicate x_ck pattern from SP1 encountered. OC block will NOT be duplicated because of no improvement. Solution run will be continued in the next iteration.")
                    log_duplicate_pattern_diagnostic(mp=mp, sp1_sol=sp1_sol, data=instance_data)
                    duplicate_oc_blocks_skipped += 1
                    # logging.info("Duplicate x_ck pattern from SP1 encountered. Skipping OC block and forcing diversification.")
                    # mp._add_no_good_cut(sp1_sol.x_ck)  # Add no-good cut to forbid this exact x_ck pattern in future iterations
                else:
                    new_pattern_this_iteration = True
                    generated_patterns_kkt_blocks.add(key)
                    generated_patterns.append(sp1_sol.x_ck)  # Store the pattern for logging and analysis
                    # Add KKT Optimality Cut to MP based on SP1 solution
                    logging.info("Adding KKT-OC block based on x_ck of SP1 solution to cut off current leader solution.")
                    if sos1_cuts:
                        mp._add_kkt_oc_block_sos1(sp1_sol.x_ck, primal_dual_streghtening=primal_dual_strenghtening)
                    else:
                        mp._add_kkt_oc_block_bigM(sp1_sol.x_ck)
                    oc_blocks_added += 1

        # ============================================================
        # Update objective cutoffs if bound_cutoff is enabled
        # ============================================================
        if not terminate:
            if bound_cutoff:
                mp.update_objective_cutoffs(
                    lower_bound=LB if lb_updated else None,
                    upper_bound=UB if ub_updated else None,
                    tolerance=cutoff_bound_tolerance
                )

        # ============================================================
        # Iteration summary
        # ============================================================
        if Verbose:
            endtime_iteration = time.perf_counter()
            iteration_time = endtime_iteration - starttime_iteration
            total_time = endtime_iteration - start_total

            logging.info("\n" + "-"*70)
            logging.info(f"End of Iteration {iteration} Summary:")
            logging.info(f"Best Incumbent MP Objective: {mp_sol.mp_obj:.5f}, MP Bound: {mp_sol.mp_bound:.5f}")
            rel_gap_str = (f"{(UB - LB) / abs(UB) * 100:.2f} %" if math.isfinite(LB) and math.isfinite(UB) and UB != 0 else 'N/A')
            logging.info(f"LB = {LB:.5f}, UB = {UB:.5f}, Gap (abs) = {(UB - LB):.5f}, Gap (relative) = {rel_gap_str}")
            if not duplicate_pattern_this_iteration:
                logging.info(f"The KKT-OC block was added based on x_ck pattern: {sp2_sol.x_ck if sp2_sol.feasible else sp1_sol.x_ck}")
            else:
                logging.info(f"Duplicate x_ck pattern encountered. No new KKT-OC block added this iteration. Continue MP exploration in next iteration.")
            logging.info(f"Total OC blocks added so far: {oc_blocks_added} (Duplicate patterns skipped: {duplicate_oc_blocks_skipped})")
            logging.info(f"Iteration time: {iteration_time:.2f} s | Total time so far: {total_time:.2f} s")
            logging.info("-"*70)

        # ============================================================
        # Carry decomposition progress information to next iteration
        # ============================================================
        previous_iteration_duplicate = duplicate_pattern_this_iteration
        previous_iteration_added_pattern = new_pattern_this_iteration
        previous_iteration_meaningful_lb_improvement = (meaningful_lb_improvement_this_iteration)

        if terminate:
            break

    # End timer for overall algorithm
    end_total = time.perf_counter()
    decomp_solution_time = end_total - start_total

    # Build DecompositionSolution object to summarize results
    decomp_sol = build_decomposition_solution(
        iterations=iteration,
        max_iterations=max_iterations,
        iteration_best_solution=iteration_best_solution,
        total_solution_time=decomp_solution_time,
        LB=LB,
        UB=UB,
        Xi=Xi,
        best_bilevel_mp_sol=best_bilevel_mp_sol,
        best_bilevel_sp2_sol=best_bilevel_sp2_sol,
        termination_reason=termination_reason,
        equality_tol=1e-3
    )
    
    #region Final Solution Summary
    def _nonzero_items(d: dict, tol: float = 1e-6):
        return [(k, v) for k, v in d.items() if abs(float(v)) > tol]

    def _log_dict(name: str, d: dict, tol: float = 1e-6) -> None:
        def _fmt_index(idx) -> str:
            if isinstance(idx, tuple):
                return "[" + ",".join(str(x) for x in idx) + "]"
            return f"({idx})"
        
        nz = _nonzero_items(d, tol)
        logging.info(f"• {name}: {len(nz)} nonzero")
        for k, v in sorted(nz):
            logging.info(f"  {name}{_fmt_index(k)} = {float(v):.10g}")

    def solution_summary(decomp_sol: DecompositionSolution, tol: float = 1e-6) -> None:
        logging.info("\n" + "#"*70)
        logging.info("DECOMPOSITION ALGORITHM - SOLUTION SUMMARY:")
        logging.info("#"*70)

        logging.info("META PARAMETERS")
        logging.info("-"*50)
        logging.info(f"Instance: {instance.instance_name} (Size class: {instance.instance_size_class}, Structural regime: {instance.instance_regime})")
        logging.info(f"Network Dimensions Total: {instance.G_max + instance.S_max + instance.I_max + instance.L_max + instance.C_max} " 
                     f"(G = {instance.G_max}, S = {instance.S_max}, I = {instance.I_max}, L = {instance.L_max}, C = {instance.C_max})"
                     )
        logging.info(f"Decomposition Parameters: Xi={Xi}, Max Iterations={max_iterations}, Soution Time Limit={total_time_limit}, MIP-Gap MasterProblem={mip_gap}")
        logging.info(f"Objective Weights: Environment={instance_data.weight_env:.2f}, Monetary={instance_data.weight_mon:.2f}")
        logging.info(f"Objective Scale: {objective_scale}")
        if sos1_cuts:
            logging.info("Complementarity Modelling: SOS1")
            logging.info(
                f"Primal-Dual Strengthening: "
                f"{'Enabled' if primal_dual_strenghtening else 'Disabled'}"
            )
        else:
            logging.info("Complementarity Modelling: Big-M")
            logging.info(
                f"Generic Dual Big-M: {bigM_duals_unrestricted:.6g}"
            )
            logging.info("Primal-Dual Strengthening: N/A")
        logging.info(f"Bound Cutoff: {f'Enabled (tolerance {cutoff_bound_tolerance})' if bound_cutoff else 'Disabled'}")
        logging.info("-"*50)

        logging.info("SOLUTION")
        logging.info(f"Status: {decomp_sol.status.name}")
        logging.info(f"Termination Reason: {decomp_sol.termination_reason}")
        logging.info(f"Iterations: {decomp_sol.iterations}/{decomp_sol.max_iterations} (OC blocks added: {oc_blocks_added}, Duplicate patterns skipped: {duplicate_oc_blocks_skipped})")
        if decomp_sol.iteration_best_solution is not None:
            logging.info(f"Best solution found in iteration: {decomp_sol.iteration_best_solution}")
        logging.info(f"Total Solution Time: {decomp_sol.total_solution_time:.2f} seconds")
        logging.info(f"Final incumbent MP Objective: {decomp_sol.best_bilevel_mp_sol.mp_obj:.5f}" if decomp_sol.best_bilevel_mp_sol is not None else "No incumbent MP solution")
        logging.info(f"Final LB (best Master): {decomp_sol.lower_bound:.5f}")
        logging.info(f"Final UB (best SP2): {decomp_sol.upper_bound:.5f}")
        logging.info(f"Final Gap (abs): {decomp_sol.final_gap_proven:.5f}" if decomp_sol.final_gap_proven is not None else "Final Gap (proven): N/A")
        logging.info(f"Final Gap (realtive): {((decomp_sol.upper_bound - decomp_sol.lower_bound) / abs(decomp_sol.upper_bound)) * 100:.2f}%" if math.isfinite(decomp_sol.lower_bound) and math.isfinite(decomp_sol.upper_bound) and decomp_sol.upper_bound != 0 else "Final Gap (relative): N/A")
        logging.info(f"MP-SP2 incumbent gap (abs): {decomp_sol.final_gap_incumbents:.5f}" if decomp_sol.final_gap_incumbents is not None else "Final Gap (incumbents): N/A")
        logging.info(f"MP-SP2 incumbent gap (relative): {((decomp_sol.upper_bound - decomp_sol.best_bilevel_mp_sol.mp_obj) / abs(decomp_sol.upper_bound)) * 100:.2f}%" if decomp_sol.best_bilevel_mp_sol is not None and math.isfinite(decomp_sol.best_bilevel_mp_sol.mp_obj) and math.isfinite(decomp_sol.upper_bound) and decomp_sol.upper_bound != 0 else "Final Gap (incumbent relative): N/A")

        logging.info(f"\nCutted patterns (x_ck fixed patterns with KKT-OC blocks added):")
        for i, pattern in enumerate(generated_patterns, start=1):
            logging.info(f" Configuration {i}: {format_pattern_dict(pattern)}")

        if decomp_sol.status is DecompositionStatus.OPTIMAL_PROVEN and decomp_sol.final_gap_proven is not None:
            logging.info(f"Final Gap (proven): {decomp_sol.final_gap_proven:.5f}")
        if decomp_sol.status is DecompositionStatus.SUBOPTIMAL_MPSP2_INCUMBENTS_MATCH and decomp_sol.final_gap_incumbents is not None:
            logging.info(f"Final Gap (incumbent solutions): {decomp_sol.final_gap_incumbents:.5f}")

        if decomp_sol.status is DecompositionStatus.MP_INFEASIBLE_OR_NO_SOLUTION:
            logging.info("No feasible solution found for Master Problem during decomposition.")
        elif decomp_sol.status is DecompositionStatus.NO_BILEVEL_FEASIBLE_SOLUTION:
            logging.info("No feasible bilevel solution found during decomposition.")
        else:
            logging.info("\n" + "#"*70)
            logging.info(f"Best bilevel solution found")
            logging.info("#"*70)

            logging.info("\nMunicipality [Leader]")
            logging.info("" + "-"*70)

            logging.info("\nObjective breakdown:\n")
            logging.info(f"Weights: Environment ={instance_data.weight_env:.2f}, Monetary={instance_data.weight_mon:.2f}")
            for index, (component, value) in enumerate(decomp_sol.best_bilevel_mp_sol.objective_components.items(), start=1):
                logging.info(f"{component:<30} {float(value):>14.6f}")
                if index in (5,10):  # Add extra spacing after transport and treatment costs for readability
                    logging.info("")
            
            leader_dict_vars = ["q_gsw", "q_slw", "q_siw", "d_siw", "z_wh", "y_wh"]
            leader_scalar_vars = ["mu_land", "mu_inc", "mu_kiln"]

            logging.info(f"\nNonzero variables in Leader Problem (|x| > {tol}):")
            for name in leader_dict_vars:
                _log_dict(name, getattr(decomp_sol.best_bilevel_mp_sol, name), tol)
            for name in leader_scalar_vars:
                val = float(getattr(decomp_sol.best_bilevel_mp_sol, name))
                if abs(val) > tol:
                    logging.info(f"{name} = {val:.10g}")

            logging.info("\nCement Producer [Follower]")
            logging.info("" + "-"*70)

            logging.info("\nObjective breakdown:\n")
            for index, (component, value) in enumerate(decomp_sol.best_bilevel_sp2_sol.follower_objective_components.items(), start=1):
                logging.info(f"{component:<45} {float(value):>14.6f}")
                if index in (4, 7):
                    logging.info("")

            if getattr(decomp_sol.best_bilevel_sp2_sol, "follower_obj_reduced", None) is not None:
                logging.info(
                    f"\nFollower objective used for decomposition (reduced): "
                    f"{decomp_sol.best_bilevel_sp2_sol.follower_obj_reduced:.6f}"
                )

            if getattr(decomp_sol.best_bilevel_sp2_sol, "follower_obj_original", None) is not None:
                logging.info(
                    f"Follower objective for reporting (reconstructed original): "
                    f"{decomp_sol.best_bilevel_sp2_sol.follower_obj_original:.6f}"
                )
            
            follower_dict_vars = ["x_ck", "q_cf", "r_sw", "q_scw"]
            logging.info(f"\nNonzero variables in Follower Problem (|x| > {tol}):")
            for name in follower_dict_vars:
                _log_dict(name, getattr(decomp_sol.best_bilevel_sp2_sol, name), tol)

    solution_summary(decomp_sol)
    #endregion
#endregion


# #region Setup Logger
# def setup_logger() -> Path:
#     """Setup logging to file and console"""
#     # Create solutions folder if it doesn't exist
#     log_dir = Path(__file__).parent.parent / "Solutions"
#     log_dir.mkdir(parents=True, exist_ok=True)
    
#     # Create log filename with date and time
#     now = datetime.now()
#     log_filename = f"Yue_KKT_Decomp_{now.strftime('%Y%m%d_%H%M')}.log"
#     log_path = log_dir / log_filename
    
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(message)s',
#         handlers=[
#             logging.FileHandler(log_path, encoding='utf-8'),
#             logging.StreamHandler()  # Also print to console
#         ]
#     )
    
#     return log_path
# #endregion

# if __name__ == "__main__":
#     log_path = setup_logger()
#     logging.info(f"Yue-KKT Decomposition Algorithm started. Logs will be saved to {log_path}")
#     run_yue_decomposition(Verbose=True)