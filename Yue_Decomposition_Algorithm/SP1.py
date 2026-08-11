import gurobipy as gp
from gurobipy import GRB
import logging

from Instances.instance_generator import InstanceData
from .MP_BigM import MasterSolution


from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class SubProblem1Solution:
    '''
    Data class to hold SP1 solution
    
    -----
    sp1_obj:
        Reduced follower objective value used for decomposition logic.
        The leader-fixed constant c_penalty * sum_{s,w} A_sw is omitted.

    sp1_obj_original:
        Reconstructed original follower objective value, including
        c_penalty * sum_{s,w} r_sw. Use for reporting only.
    '''
    sp1_obj: float
    sp1_obj_original: float
    x_ck: Dict[Tuple[int, int], float]          # Investment decision of capacity k for cement plant c
    q_cf: Dict[Tuple[int, int], float]          # Quantity of coal f processed at cement plant c
    r_sw: Dict[Tuple[int, int], float]          # Residual waste w at transfer station s after allocation (not utilized by cement plants)
    q_scw: Dict[Tuple[int, int, int], float]    # Quantity of waste w from transfer station s to cement plant c
    objective_components: Dict[str, float] | None = None         # Optional dictionary to hold detailed objective cost components for reporting


class SubProblem1:
    """Subproblem 1: Follower Optimal reaction Problem
    
    This class models the subproblem (SP1) in the bilevel optimization framework.
    It determines the followers' optimal response for fixed leader decisions x_L*, y_L* from the master problem (MP)
    
    The follower (cement producer) determines:
    - Investment decision for pre- & co-processing capacity expansion at cement plants
    - Quantity of waste and coal processed at cement plants
    - Quantity of waste flow from transfer stations to cement plants (if allocated)
    - Implicitly: Residual waste (denied quantity) at transfer stations after allocation to cement plants

    In the reduced follower formulation, residual waste r_sw is no longer an
    explicit decision variable. Instead, it is reconstructed from
        r_sw = A_sw - sum_c q_scw,
    where
        A_sw = sum_g q_gsw - sum_l q_slw - sum_i q_siw.
    
    Decision variables:
    - x_ck: Binary variable for investment decision of capacity k for cement plant c
    - q_cf: Continuous variable for quantity of coal f processed at cement plant c
    - q_scw: Continuous variable for quantity of waste w from transfer station s to cement plant c (if allocated)

    Reconstructed quantities (affine expression):
    - r_sw: residual waste w at transfer station s after allocation (not utilized and denied by cement plants)
    """
    
    def __init__(self, instance: InstanceData):
        """Initialize SP1 with instance data"""
        self.instance = instance
        self._build = False
        self.model = None

        # Variable Containers (filled during build)
        # Follower variables
        self.x_ck = None
        self.q_cf = None
        self.q_scw = None

        # fixed leader decisions from MP solution (input to SP1, filled in build() method)
        self.mu_kiln = None
        self.z_wh = None
        self.q_gsw = None
        self.q_slw = None
        self.q_siw = None

        # Objective expressions for reporting
        self.obj_cost_coal = None
        self.obj_cost_invest = None
        self.obj_cost_preproc = None
        self.obj_cost_transport = None
        self.obj_revenue_subsidy = None

        # Reduced penalty contribution: -c_penalty * sum q_scw
        self.obj_penalty_reduced_contribution = None

        # Original reconstructed penalty: c_penalty * sum r_sw
        self.obj_cost_penalty_reconstructed = None

        self.obj_total_follower_reduced = None
        self.obj_total_follower_original = None

    #region Station Availability
    def _availability_A_sw(self, s: int, w: int, *, tol: float = 1e-6) -> float:            # 1e-6 because FeasibilityTol = 1e-6 in MP
        """Compute leader-induced station availability A_sw.

        A_sw = sum_g q_gsw - sum_l q_slw - sum_i q_siw.

        In the reduced follower model, A_sw is fixed in SP1 because all leader
        variables are fixed at the MP solution.
        """
        value = (
            sum(self.q_gsw[g, s, w] for g in self.instance.G)
            - sum(self.q_slw[s, l, w] for l in self.instance.L)
            - sum(self.q_siw[s, i, w] for i in self.instance.I)
        )

        # Clean tiny numerical noise, but do not hide structural infeasibility.
        if abs(value) <= tol:
            return 0.0

        return float(value)
    #endregion

    #region Compute Residue
    def _reconstruct_r_sw(self, *, tol: float = 1e-6) -> Dict[Tuple[int, int], float]:
        """Reconstruct residual waste after solving SP1.

        r_sw = A_sw - sum_c q_scw.

        The variable r_sw is not part of the reduced optimization model anymore.
        It is reconstructed for reporting and for compatibility with downstream code.
        """
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError("Cannot reconstruct r_sw because SP1 has no solution.")

        data = self.instance
        residuals: Dict[Tuple[int, int], float] = {}

        for s in data.S:
            for w in data.W:
                accepted = sum(self.q_scw[s, c, w].X for c in data.C)
                residual = self._availability_A_sw(s, w) - accepted

                if residual < -tol:
                    logging.warning(
                        # f"Reconstructed residual r_sw[{s},{w}] is negative: {residual:.6g}."
                        "Reconstructed residual r_sw[%s,%s] is negative: %.6g. "
                        "This indicates numerical tolerance issues or an inconsistent leader solution.",
                        s, w, residual
                    )

                residuals[(s, w)] = 0.0 if abs(residual) <= tol else float(residual)

        return residuals
    #endregion
    
    #region Build Model
    def build(self, mp_sol: MasterSolution, *, name: str = "SubProblem1", output_flag: int = 1) -> None:
        if self._build: 
            raise RuntimeError("SP1 model was already built.")
        
        """Build the Subproblem 1 model"""
        self.model = gp.Model(name)
        self.model.setParam('OutputFlag', output_flag)
        
        #  ---- fixed leader decisions from MP solution (input) ----
        # These are the decisions from the master problem that are fixed in SP1
        
        # clean data dictionaries from numerical noise (e.g., 1e-10 instead of 0) to avoid numerical issues in SP1 and improve performance
        def clean_continuous_value(x: float, tol: float = 1e-6) -> float:
            return 0.0 if abs(x) <= tol else float(x)

        def clean_continuous_dict(d: dict, tol: float = 1e-6) -> dict:
            return {k: clean_continuous_value(v, tol) for k, v in d.items()}
        
        self.mu_kiln = clean_continuous_value(mp_sol.mu_kiln)
        self.z_wh = mp_sol.z_wh     # binary variable, no need to clean because already rounded in MP solution
        self.q_gsw = clean_continuous_dict(mp_sol.q_gsw)
        self.q_slw = clean_continuous_dict(mp_sol.q_slw)
        self.q_siw = clean_continuous_dict(mp_sol.q_siw)

        #  ---- add variables / constraints / objective ----
        self._add_variables()
        self._add_constraints()
        self._set_objective()
        self.model.update()

        logging.info("\nSubproblem 1 model structure:\n")
        logging.info(f"  → Total created variables: {self.model.NumVars}")
        logging.info(f"  → Thereof binary variables: {self.model.NumBinVars}")
        logging.info(f"  → Thereof continuous variables: {self.model.NumVars - self.model.NumBinVars}")

        logging.info(f"  → Total created constraints: {self.model.NumConstrs}\n")

        self._build = True
    #endregion

    #region Solve Model
    def solve(self, *, time_limit: int = GRB.INFINITY) -> None:
        assert self.model is not None, "Model is not built yet. Call build() before solve()."
        self.model.Params.TimeLimit = time_limit
        
        """Solve the Subproblem 1"""
        logging.info("\n" + "-"*60)
        logging.info("Solving Subproblem 1...")
        logging.info(f"  → Time limit: {time_limit} seconds")
        logging.info("-"*60)

        self.model.Params.NumericFocus = 2  # Focus on numerical issues to improve solution reliability for SP1
        self.model.Params.IntFeasTol = 1e-8
        self.model.Params.FeasibilityTol = 1e-8
        self.model.Params.IntegralityFocus = 1
        self.model.Params.MIPGap = 1e-6
        self.model.optimize()

        if self.model.Status == GRB.INF_OR_UNBD:
                    logging.info("SP1 is infeasible or unbounded. Re-solving with DualReductions=0 to obtain conclusion...")
                    self.model.Params.DualReductions = 0
                    self.model.reset()
                    self.model.optimize()
    #endregion

    #region Extract Solution
    def extract_solution(self) -> SubProblem1Solution:
        """Extract solution from the Master Problem"""
        if self.model.status == GRB.OPTIMAL:
            logging.info('✓ Subproblem 1 solved optimally.')
        # elif self.model.status == GRB.SUBOPTIMAL:
        #     logging.info('⚠ Subproblem 1 solved suboptimally.')
        # elif self.model.status == GRB.TIME_LIMIT and self.model.SolCount > 0:
        #     logging.info('⚠ Subproblem 1 solve time limit reached. Best solution found will be extracted.')
        elif self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write(f"SP1_infeasible.ilp")
            raise RuntimeError("✗ Subproblem 1 is infeasible. No solution can be extracted.")
        elif self.model.Status == GRB.UNBOUNDED:
            self.model.write("SP1_unbounded.lp")
            raise RuntimeError("✗ Subproblem 1 is unbounded. No solution can be extracted.")
        else:
            # raise RuntimeError("✗ No solution for Subproblem 1 found; cannot extract solution.")
            raise RuntimeError(f"✗ Subproblem 1 not solved to optimality. Status={self.model.status}. "
                               f"SolCount={self.model.SolCount}, Gap={getattr(self.model, 'MIPGap', None)}.\n"
                               f"No solution for Subproblem 1 can be extracted, because optimum is required for decomposition logic.")
        
        data = self.instance
        r_sw_reconstructed = self._reconstruct_r_sw()

        objective_components = {
            "Coal cost": float(self.obj_cost_coal.getValue()),
            "Investment cost": float(self.obj_cost_invest.getValue()),
            "Pre-processing cost": float(self.obj_cost_preproc.getValue()),
            "Transport cost to kilns": float(self.obj_cost_transport.getValue()),
            "Reduced penalty contribution": float(
                self.obj_penalty_reduced_contribution.getValue()
            ),
            "Reconstructed original penalty cost": float(
                self.obj_cost_penalty_reconstructed.getValue()
            ),
            "Subsidies received": float(self.obj_revenue_subsidy.getValue()),
            "Reduced follower objective": float(
                self.obj_total_follower_reduced.getValue()
            ),
            "Original follower objective reconstructed": float(
                self.obj_total_follower_original.getValue()
            ),
        }

        return SubProblem1Solution(
            sp1_obj=self.model.ObjVal,      # self.obj_total_follower_reduced.getValue() would be the same
            sp1_obj_original=self.obj_total_follower_original.getValue(),
            x_ck={(c, k): int(round(self.x_ck[c, k].X)) for c in data.C for k in data.K},       # rounding because of floating-point relaxation within gurobi (0.9999997 or 1.0000002 possible)
            q_cf={(c, f): self.q_cf[c, f].X for c in data.C for f in data.F},
            r_sw=r_sw_reconstructed,
            q_scw={(s, c, w): self.q_scw[s, c, w].X for s in data.S for c in data.C for w in data.W},
            objective_components=objective_components
        )
    #endregion
    
    #region Add Variables
    def _add_variables(self) -> None:
        """Add decision variables for SP1"""
        data = self.instance
        m = self.model

        self.x_ck = m.addVars(data.C, data.K, vtype=GRB.BINARY, name="x_ck")                # Investment decision of capacity k for cement plant c
        self.q_cf = m.addVars(data.C, data.F, lb=0.0,vtype=GRB.CONTINUOUS, name="q_cf")            # Quantity of coal f processed at cement plant c
        self.q_scw = m.addVars(data.S, data.C, data.W, lb=0.0,vtype=GRB.CONTINUOUS, name="q_scw")  # Quantity of waste w from transfer station s to cement plant c (if allocated)
    #endregion

    #region Follower Constraints
    def _add_constraints(self) -> None:
        """Add constraints for SP1 (equal to the follower's problem constraints with fixed leader decisions)"""
        data = self.instance
        m = self.model

        # (F1) # Only one pre- and co-processing capacity per cement facility feasible
        m.addConstrs(
            (gp.quicksum(self.x_ck[c,k] for k in data.K) <= 1 for c in data.C),
        name="F1_capacityChoice"
        )

        # (F2) Cement facility budget constraint for investing in pre- & co-processing
        # scale numerics for better matrix and RHS vaulues, but keep feasible set untouched (e.g., if costs are in the order of 10,000 and budget is 1,000,000, scale down costs by factor of 1000 to get values in the order of 10 and keep budget at 1000)
        budget_cem_scale = 1_000_000
        m.addConstr(
            gp.quicksum(self.x_ck[c,k]* (data.c_invest_k[k] / budget_cem_scale) for c in data.C for k in data.K) <= data.budget_cem / budget_cem_scale,
        name="F2_cementBudget"
        )

        # (F3) Energy fulfillment in cement kiln
        m.addConstrs(
            (gp.quicksum(self.q_cf[c,f]*data.beta_f[f] for f in data.F) + gp.quicksum(self.q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) >= data.alpha_c[c] for c in data.C),
        name="F3_energyFulfillment"
        )

        # (F4) Co-processing capacity limitation
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) <= data.kappa_coproc*data.alpha_c[c] for c in data.C),
        name="F4_coprocCapacityLimit"
        )

        # (F5) Pre- & co-processing capacity according to investment decision
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w] for s in data.S for w in data.W) <= gp.quicksum(self.x_ck[c,k]*data.Q_k[k] for k in data.K) for c in data.C),
        name="F5_investmentCapacity"
        )

        # (F6) Station-wise availability constraint
        # sum_c q_scw <= A_sw
        #
        # This replaces both:
        # - old F6 quota equality
        # - old F7 station balance equality
        #
        # Residue is reconstructed after solving:
        # r_sw = A_sw - sum_c q_scw
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w] for c in data.C) <= self._availability_A_sw(s, w) for s in data.S for w in data.W),
        name="F6_stationAvailability",
        )
    #endregion

    #region Set Objective
    def _set_objective(self) -> None:
        """
        Set objective function for SP1 (follower's problem objective)
        
        The original penalty term
            c_penalty * sum_{s,w} r_sw
        is replaced by
            -c_penalty * sum_{s,c,w} q_scw
        after dropping the leader-fixed constant c_penalty * sum_{s,w} A_sw.

        Therefore, the optimized SP1 objective is the reduced objective used for
        decomposition. The original objective is reconstructed for reporting.
        """
        data = self.instance
        m = self.model

        # ---- Define auxiliary variables for cost components ----
        # 1) Cost of coal
        self.obj_cost_coal = gp.quicksum(self.q_cf[c,f]*data.price_f[f] for c in data.C for f in data.F)
        # 2) Investment cost
        self.obj_cost_invest = gp.quicksum(data.fixcost_invest_k[k]*self.x_ck[c,k] for c in data.C for k in data.K)
        # 3) Pre-processing cost
        self.obj_cost_preproc = gp.quicksum(data.c_preproc_w[w]*self.q_scw[s,c,w] for s in data.S for c in data.C for w in data.W)
        # 4) Transportation cost
        self.obj_cost_transport = gp.quicksum(self.q_scw[s,c,w]*data.c_truck*data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
        # 5) Reduced Penalty contribution (with leader-fixed constant dropped): -c_penalty * sum_{s,c,w} q_scw
        self.obj_penalty_reduced_contribution = data.c_penalty * gp.quicksum(self.q_scw[s,c,w] for s in data.S for c in data.C for w in data.W)
        # 6) Subsidy revenue
        self.obj_revenue_subsidy = gp.quicksum(self.q_scw[s,c,w]*gp.quicksum(data.phi_wh[w][h]*self.z_wh[w,h] for h in data.H) for s in data.S for c in data.C for w in data.W)

        # Reduced follower objective used by SP1 and decomposition logic
        self.obj_total_follower_reduced = (
            self.obj_cost_coal
            + self.obj_cost_invest
            + self.obj_cost_preproc
            + self.obj_cost_transport
            - self.obj_penalty_reduced_contribution
            - self.obj_revenue_subsidy
        )

        # Original reconstructed penalty for reporting:
        # c_penalty * sum_{s,w} (A_sw - sum_c q_scw)
        self.obj_cost_penalty_reconstructed = data.c_penalty * gp.quicksum(self._availability_A_sw(s, w) - gp.quicksum(self.q_scw[s, c, w] for c in data.C)
            for s in data.S
            for w in data.W
        )

        # Original follower objective for reporting only.
        # This differs from the reduced objective by the leader-fixed constant
        #   c_penalty * sum_{s,w} A_sw.
        self.obj_total_follower_original = (
            self.obj_cost_coal
            + self.obj_cost_invest
            + self.obj_cost_preproc
            + self.obj_cost_transport
            + self.obj_cost_penalty_reconstructed
            - self.obj_revenue_subsidy
        )

        m.setObjective(self.obj_total_follower_reduced, GRB.MINIMIZE)
    #endregion