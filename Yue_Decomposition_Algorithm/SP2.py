import gurobipy as gp
from gurobipy import GRB
import logging

from Instances.instance_generator import InstanceData
from .MP_BigM import MasterSolution
from .SP1 import SubProblem1Solution


from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class SubProblem2Solution:
    '''Data class to hold Subproblem 2 solution
    -----
    sp2_obj:
        Leader objective value, as before.

    follower_obj_reduced:
        Reduced follower objective used in the SP2 optimality constraint.

    follower_obj_original:
        Reconstructed original follower objective, including
        c_penalty * sum_{s,w} r_sw. Use for reporting only.
    '''
    feasible: bool                                     # Indicates if SP2 is feasible (followers' reaction from SP1 is feasible for the given leader decisions)

    # Leader objective
    sp2_obj: float | None                              # Objective value of SP2 (equals objective of leader)
    leader_objective_components: dict[str, float] | None  # Optional dictionary to hold the components of the leader objective function for posterior analysis (e.g., total emissions, total costs, etc.)

    #Follower objective
    follower_obj_reduced: float | None                 # Objective value of the reduced follower problem (used in SP2 optimality constraint)
    follower_obj_original: float | None                # Reconstructed original follower objective value (for reporting only, includes penalty cost of residual waste)
    follower_objective_components: dict[str, float] | None      # Optional dictionary to hold the components of the follower objective function for posterior analysis (e.g., coal cost, investment cost, subsidy revenue, etc.)

    # Follower variables
    x_ck: Dict[Tuple[int, int], float] | None          # Investment decision of capacity k for cement plant c
    q_cf: Dict[Tuple[int, int], float] | None          # Quantity of coal f processed at cement plant c
    r_sw: Dict[Tuple[int, int], float] | None          # Residual waste w at transfer station s after allocation (reconstructed, not optimized)
    q_scw: Dict[Tuple[int, int, int], float] | None    # Quantity of waste w from transfer station s to cement plant c
    

class SubProblem2:
    '''
    Subproblem 2: Follower feasibility problem

    This class models the subproblem (SP2) in the bilevel optimization framework of Yue et al. (2017). 
    It takes the leaders' optimal decisions from the master problem (MP) and optimal followers reaction value from Subproblem 1 (SP1) to check
    the feasibility of the followers reaction by looking for lower level variables satisfying all constraints (leader and follower) while achieving
    an objective value at least as good as the optimal reaction value from SP1. If SP2 is infeasible, it indicates that the followers' reaction 
    from SP1 is not feasible for the given leader decisions (leader decision is inconsistent with rational follower behaviour), and we can generate a 
    feasibility cut for the master problem. If SP2 is feasible, it confirms that the followers' reaction from SP1 is indeed feasible for the given 
    leader decisions, and we can proceed to generate an optimality cut based on the dual information from SP2. 
    '''

    def __init__(self, instance: InstanceData):
        '''Initialize SP2 with instance data'''
        self.instance = instance
        self._build = False
        self.model = None

        # Variable containers (filled during build() )
        # Follower variables
        self.x_ck = None  # Investment decision of capacity k for cement plant c
        self.q_cf = None  # Quantity of coal f processed at cement plant c
        self.q_scw = None  # Quantity of waste w from transfer station s to cement plant c

        # fixed leader decisions from MP solution (input to SP1, filled in build() method)
        self.mu_kiln = None
        self.z_wh = None
        self.q_gsw = None
        self.q_slw = None
        self.q_siw = None
        self.d_siw = None

        # obtained optimal reaction value from SP1 (input to SP2, filled in build() method)
        # IMPORTANT: this must be the reduced SP1 objective value, because SP1 optimizes the reduced expression
        self.sp1_optimal_value = None

        # Objective function components of lower level problem for posterior analysis
        self.obj_cost_coal = None
        self.obj_cost_invest = None
        self.obj_cost_preproc = None
        self.obj_penalty_reduced_contribution = None
        self.obj_cost_penalty_reconstructed = None
        self.obj_cost_transport = None
        self.obj_revenue_subsidy = None
        self.obj_total_follower_reduced = None
        self.obj_total_follower_original = None

        # Objective function components of leader (SP2 objective)
        self.obj_emission_transport_leader = None
        self.obj_emission_treatment_leader = None
        self.obj_emission_fuel_leader = None

        self.obj_cost_transport_leader = None
        self.obj_cost_treatment_leader = None
        self.obj_cost_subsidy_leader = None
        
        self.obj_total_env_leader = None
        self.obj_total_env_weighted_leader = None

        self.obj_total_mon_leader = None
        self.obj_total_mon_weighted_leader = None

    #region Helper functions
    #region Clean nonnegatives
    def _clean_nonnegative(self, value: float, *, tol: float = 1e-6) -> float:
        """Clean small numerical noise in nonnegative values.

        If the value is within the tolerance of zero, return zero.
        Otherwise, return the original value.

        """
        value = float(value)

        if -tol <= value < 0.0:
            return 0.0
        return value
    #endregion
    
    #region Auxiliary A_sw
    def _availability_A_sw(self, s: int, w: int, *, tol: float = 1e-6) -> float:
        """Compute leader-induced station availability A_sw.

        A_sw = sum_g q_gsw - sum_l q_slw - sum_i q_siw.

        In the reduced follower model, A_sw is fixed because all leader
        variables are fixed at the MP solution.
        """
        value = (
            sum(self.q_gsw[g, s, w] for g in self.instance.G)
            - sum(self.q_slw[s, l, w] for l in self.instance.L)
            - sum(self.q_siw[s, i, w] for i in self.instance.I)
        )

        # Clean tiny numerical noise, but do not hide structural infeasibility.
        # if abs(value) <= tol:
        #     return 0.0

        # return float(value)
        return self._clean_nonnegative(value, tol=tol)
    #endregion

    #region Clean Required Co
    def _required_coprocessing_sw(
        self,
        s: int,
        w: int,
        *,
        tol: float = 1e-6
    ) -> float:
        """
        Required co-processing implied by the fixed MP leader solution:
            sum_c q_scw[s,c,w]
            = A_sw[s,w] - sum_i d_siw[s,i,w].

        This quantity is theoretically nonnegative.
        """

        value = (self._availability_A_sw(s, w, tol=tol) - sum(self.d_siw[s, i, w] for i in self.instance.I))

        return self._clean_nonnegative(value, tol=tol)
    #endregion
    
    #region Construct r_sw
    def _reconstruct_r_sw(self, *, tol: float = 1e-7) -> Dict[Tuple[int, int], float]:
        """Reconstruct residual waste after solving SP2.

        r_sw = A_sw - sum_c q_scw.

        The variable r_sw is not part of the reduced SP2 model.
        """
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError("Cannot reconstruct r_sw because SP2 has no solution.")

        data = self.instance
        residuals: Dict[Tuple[int, int], float] = {}

        for s in data.S:
            for w in data.W:
                accepted = sum(self.q_scw[s, c, w].X for c in data.C)
                residual = self._availability_A_sw(s, w) - accepted

                if residual < -tol:
                    logging.warning(
                        f"Reconstructed residual r_sw[{s},{w}] is negative: {residual:.6g}. "
                        f"This indicates numerical tolerance issues or an inconsistent leader solution."
                    )

                residuals[(s, w)] = 0.0 if abs(residual) <= tol else float(residual)

        return residuals
    #endregion
    #endregion

    #region Build model
    def build(self, mp_solution: MasterSolution, sp1_solution: SubProblem1Solution, *, name: str = "SubProblem2", output_flag: int = 0, objective_scale: float = 1.0) -> None:
        '''Build the Subproblem 2 model '''
        if self._build:
            raise RuntimeError("SP2 was already built.")
        
        self.model = gp.Model(name)
        self.model.setParam('OutputFlag', output_flag)

        # Extract fixed leader decisions from MP solution

        # clean data dictionaries from numerical noise (e.g., 1e-10 instead of 0) to avoid numerical issues in SP1 and improve performance
        def clean_continuous_value(x: float, tol: float = 1e-9) -> float:
            return 0.0 if abs(x) <= tol else float(x)

        def clean_continuous_dict(d: dict, tol: float = 1e-9) -> dict:
            return {k: clean_continuous_value(v, tol) for k, v in d.items()}
        
        self.mu_kiln = clean_continuous_value(mp_solution.mu_kiln)
        self.mu_land = clean_continuous_value(mp_solution.mu_land)
        self.mu_inc = clean_continuous_value(mp_solution.mu_inc)
        self.z_wh = mp_solution.z_wh  # binary variable, no need to clean because already rounded in MP solution
        self.q_gsw = clean_continuous_dict(mp_solution.q_gsw)
        self.q_slw = clean_continuous_dict(mp_solution.q_slw)
        self.q_siw = clean_continuous_dict(mp_solution.q_siw)
        self.d_siw = clean_continuous_dict(mp_solution.d_siw)

        # Extract optimal reaction value from SP1 solution
        # This must be the reduced follower objective value, because SP2 uses
        # the reduced follower objective in the optimality constraint.
        self.sp1_optimal_value = sp1_solution.sp1_obj

        #  ---- add variables / constraints / objective ----
        self._add_variables()
        self._add_constraints()
        self._set_objective(objective_scale=objective_scale)
        # self.model.setObjective(0.0, GRB.MINIMIZE)  # Objective is zero because we are only checking feasibility of achieving SP1 optimal reaction value with the given leader decisions (SP2 is a feasibility problem)
        self.model.update()

        logging.info("\nSubproblem 2 model structure:\n")
        logging.info(f"  → Total created variables: {self.model.NumVars}")
        logging.info(f"  → Thereof binary variables: {self.model.NumBinVars}")
        logging.info(f"  → Thereof continuous variables: {self.model.NumVars - self.model.NumBinVars}")

        logging.info(f"  → Total created constraints: {self.model.NumConstrs}\n")

        self._build = True
    #endregion

    #region Solve model
    def solve(self, *, time_limit: int = GRB.INFINITY) -> None:
        '''Solve SP2'''
        assert self.model is not None, "Model is not built yet. Call build() before solve()."
        self.model.Params.TimeLimit = time_limit

        """Solve the Subproblem 2"""
        logging.info("\n" + "-"*60)
        logging.info("Solving Subproblem 2...")
        logging.info(f"  → Time limit: {time_limit} seconds")
        logging.info("-"*60)

        self.model.Params.NumericFocus = 2  # Focus on numerical issues to improve solution reliability for SP2, which is a feasibility problem and can be more sensitive to numerical issues
        self.model.Params.IntFeasTol = 1e-8
        self.model.Params.FeasibilityTol = 1e-6
        self.model.Params.IntegralityFocus = 1
        self.model.Params.MIPGap = 1e-6
        self.model.optimize()
    #endregion

    def _set_gurobi_parameters(self) -> None:           # only needed if feasibility problem considered with 0 objective, otherwise not necessary to set special parameters for optimality focus
        '''Set some specific parameters for solution run'''
        self.model.setParam('MIPFocus', 1)  # Focus on finding a feasible solution
        self.model.Params.SolutionLimit = 1  # Stop after finding the first feasible solution (if any)
        # self.model.NoRelHeurTime = 300  # Heuristic to find feasible solutions (if any) for up to 5 minutes

    #region Get objective
    def get_objective_components(self) -> dict[str, float]:
        '''Evaluate follower objective components at current incubent solution for posterior analysis'''
        if self.model is None:
            raise RuntimeError("Model is not built yet. Call build() before getting objective components.")
        elif self.model.SolCount == 0:
            raise RuntimeError("SP2 is infeasible / has no solution. Cannot get objective components.")
        
        return {
            "Coal cost": float(self.obj_cost_coal.getValue()),
            "Investment cost": float(self.obj_cost_invest.getValue()),
            "Pre-processing cost": float(self.obj_cost_preproc.getValue()),
            "Transportation cost": float(self.obj_cost_transport.getValue()),
            "Penalty cost (reduced)": float(self.obj_penalty_reduced_contribution.getValue()),
            "Penalty cost (reconstructed)": float(self.obj_cost_penalty_reconstructed.getValue()),
            "Subsidies received": float(self.obj_revenue_subsidy.getValue()),
            "Follower objective (reduced)": float(self.obj_total_follower_reduced.getValue()),
            "Follower objective (original)": float(self.obj_total_follower_original.getValue())
        }
    #endregion
    
    #region Extract solution
    def extract_solution(self) -> SubProblem2Solution:
        data = self.instance
        is_feasible = self.model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]
        is_time_limit_reached = self.model.status == GRB.TIME_LIMIT
        
        def _build_feasible_solution() -> SubProblem2Solution:
            leader_obj_components = self.get_leader_objective_components()
            follower_obj_components = self.get_objective_components()
            r_sw_reconstructed = self._reconstruct_r_sw()

            return SubProblem2Solution(
                feasible=True,

                sp2_obj=self.model.ObjVal,
                leader_objective_components=leader_obj_components,

                follower_obj_reduced=self.obj_total_follower_reduced.getValue(),
                follower_obj_original=self.obj_total_follower_original.getValue(),
                follower_objective_components=follower_obj_components,

                x_ck = {(c,k): int(round(self.x_ck[c, k].X)) for c in data.C for k in data.K},       # rounding because of floating-point relaxation within gurobi (0.9999997 or 1.0000002 possible)
                q_cf = {(c,f): self.q_cf[c, f].X for c in data.C for f in data.F},
                r_sw = r_sw_reconstructed,
                q_scw = {(s,c,w): self.q_scw[s, c, w].X for s in data.S for c in data.C for w in data.W},
            )

        if is_feasible:
            if self.model.status == GRB.OPTIMAL:
                logging.info('✓ Subproblem 2 solved optimally.')
            else:
                logging.info('⚠ Subproblem 2 solved suboptimally.')
            
            return  _build_feasible_solution()
        
        elif is_time_limit_reached and self.model.SolCount > 0:
            logging.info('⚠ Subproblem 2 solve time limit reached. Best solution found will be extracted.')
            
            return _build_feasible_solution()
        
        else:       # alternative: just 'return None' and check in decomposition algorithm 'if: sp2_sol = None'
            logging.info('✗ Subproblem 2 is infeasible.')
            return SubProblem2Solution(
                feasible=False,

                sp2_obj=None,
                leader_objective_components=None,

                follower_obj_reduced=None,
                follower_obj_original=None,
                follower_objective_components=None,

                x_ck=None,
                q_cf=None,
                r_sw=None,
                q_scw=None,
            )
    #endregion

    #region Add variables
    def _add_variables(self) -> None:
        '''Add decision variables for SP2'''
        data = self.instance
        m = self.model

        # Follower variables
        self.x_ck = m.addVars(data.C, data.K, vtype=GRB.BINARY, name="x_ck")
        self.q_cf = m.addVars(data.C, data.F, lb=0.0, vtype=GRB.CONTINUOUS, name="q_cf")
        self.q_scw = m.addVars(data.S, data.C, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="q_scw")
    #endregion

    def _add_constraints(self) -> None:
        '''Add constraints for SP2 (leader constraints, follower constraints, and the optimality constraint to ensure SP1 optimal reaction value is achieved)'''
        data = self.instance
        m = self.model

        # ==================================================== 
        # Leader constraints 
        # ====================================================
        # region Leader Constraints
        
        # ---
        # The leader constraints (L1)-(L3), (L5)-(L8), and (L10)-(L11) are not needed in SP2 because the leader decisions are fixed 
        # based on the MP solution and already satisfy these constraints by construction (the MP solution is feasible for the MP 
        # constraints, which include all leader constraints).
        # ---

        # Coupling constraint:
        # (L4) Residue waste routing euqals declined waste from cement kilns that must be incinerated instead (coupling constraint between landfill/incinerator routing and cement kiln decisions)
        # residue routed to incinerators must equal reconstructed declined waste.
        #   sum_i d_siw = A_sw - sum_c q_scw
        # d_siw is fixed from the MP solution; q_scw is chosen in SP2.
        # m.addConstrs(
        #     (gp.quicksum(self.d_siw[s,i,w] for i in data.I) 
        #      == self._availability_A_sw(s, w) - gp.quicksum(self.q_scw[s,c,w] for c in data.C) for s in data.S for w in data.W),
        # name="L4_residueWasteRouting"
        # )
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w] for c in data.C) == self._required_coprocessing_sw(s, w) for s in data.S for w in data.W),
        name="L4_residueWasteRouting"
        )

        # Coupling constraint
        # (L10) Total subsidy cost cannot exceed municipality budget
        # scale for better numerics o matrix an RHS, but keep feasible set untouched (e.g., if costs are in the order of 10,000 and budget is 1,000,000, scale down costs by factor of 1000 to get values in the order of 10 and keep budget at 1000)
        budget_mun_scale = 10_000
        m.addConstr(
            gp.quicksum(self.q_scw[s,c,w] * gp.quicksum((data.phi_wh[w][h] / budget_mun_scale) * self.z_wh[w,h] for h in data.H) for s in data.S for c in data.C for w in data.W) 
            <= data.budget_municipality / budget_mun_scale,
        name="L10_municipalityBudget"
        )
        
        #endregion


        # ====================================================
        # Follower constraints
        # ====================================================
        # region Follower Constraints

        # (F1) # Only one pre- and co-processing capacity per cement facility feasible
        m.addConstrs(
            (gp.quicksum(self.x_ck[c,k] for k in data.K) <= 1 for c in data.C),
        name="F1_capacityChoice"
        )

        # (F2) Cement facility budget constraint for investing in pre- & co-processing
        # scale numerics for better matrix and RHS vaulues, but keep feasible set untouched (e.g., if costs are in the order of 10,000 and budget is 1,000,000, scale down costs by factor of 1000 to get values in the order of 10 and keep budget at 1000)
        budget_cem_scale = 1_000_000
        m.addConstr(
            gp.quicksum(self.x_ck[c,k]* (data.c_invest_k[k] / budget_cem_scale) for c in data.C for k in data.K) 
            <= data.budget_cem / budget_cem_scale,
        name="F2_cementBudget"
        )

        # (F3) Energy fulfillment in cement kiln
        m.addConstrs(
            (gp.quicksum(self.q_cf[c,f]*data.beta_f[f] for f in data.F) 
             + gp.quicksum(self.q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) 
             >= data.alpha_c[c] for c in data.C),
        name="F3_energyFulfillment"
        )

        # (F4) Co-processing capacity limitation
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) 
             <= data.kappa_coproc*data.alpha_c[c] for c in data.C),
        name="F4_coprocCapacityLimit"
        )

        # (F5) Pre- & co-processing capacity according to investment decision
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w] for s in data.S for w in data.W) 
             <= gp.quicksum(self.x_ck[c,k]*data.Q_k[k] for k in data.K) for c in data.C),
        name="F5_investmentCapacity"
        )

        # (F6) Station-wise availability constraint
        # sum_c q_scw <= A_sw, where A_sw = sum_g q_gsw - sum_l q_slw - sum_i q_siw (leader-induced availability at station s for waste w)
        m.addConstrs(
            (gp.quicksum(self.q_scw[s,c,w] for c in data.C) 
            <= self._availability_A_sw(s, w) for s in data.S for w in data.W),
            # gp.quicksum(self.q_gsw[g,s,w] for g in data.G)
            # - gp.quicksum(self.q_slw[s,l,w] for l in data.L)
            # - gp.quicksum(self.q_siw[s,i,w] for i in data.I)
        name="F6_stationAvailability"
        )

        #endregion

        # ====================================================
        # Optimality constraint to ensure that SP2 achieves the SP1 optimal reaction value
        # ====================================================
        # region Optimality constraint

        # 1) Cost of coal
        self.obj_cost_coal = gp.quicksum(self.q_cf[c,f] * data.price_f[f] for c in data.C for f in data.F)
        # 2) Investment cost
        self.obj_cost_invest = gp.quicksum(data.fixcost_invest_k[k] * self.x_ck[c,k] for c in data.C for k in data.K)
        # 3) Pre-processing cost
        self.obj_cost_preproc = gp.quicksum(data.c_preproc_w[w] * self.q_scw[s,c,w] for s in data.S for c in data.C for w in data.W)
        # 4) Transportation cost to kilns
        self.obj_cost_transport =  gp.quicksum(self.q_scw[s,c,w] * data.c_truck * data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
        # 5) Reduced Penalty cost
        # -c_penalty * sum q_scw
        self.obj_penalty_reduced_contribution = data.c_penalty * gp.quicksum(self.q_scw[s, c, w] for s in data.S for c in data.C for w in data.W)
        # 6) Subsidy revenue
        # revenue_subsidy = gp.quicksum(data.phi_wh[w][h] * self.y_cwh[c,w,h] for c in data.C for w in data.W for h in data.H)
        self.obj_revenue_subsidy = gp.quicksum(self.q_scw[s,c,w] * gp.quicksum(data.phi_wh[w][h] * self.z_wh[w,h] for h in data.H) for s in data.S for c in data.C for w in data.W)  # equivalent formulation based on subsidy level choice z_wh

        # Reduced follower objective used for SP2 optimality
        self.obj_total_follower_reduced = (
            self.obj_cost_coal
            + self.obj_cost_invest
            + self.obj_cost_preproc
            + self.obj_cost_transport
            - self.obj_penalty_reduced_contribution
            - self.obj_revenue_subsidy
        )

        # Original reconstructed penalty for reporting only (not used in SP2 optimality constraint because SP1 optimizes the reduced expression)
        self.obj_cost_penalty_reconstructed = data.c_penalty * gp.quicksum(self._availability_A_sw(s, w) - gp.quicksum(self.q_scw[s, c, w] for c in data.C) for s in data.S for w in data.W)

        # Original follower objective for reporting
        self.obj_total_follower_original = (
            self.obj_cost_coal
            + self.obj_cost_invest
            + self.obj_cost_preproc
            + self.obj_cost_transport
            + self.obj_cost_penalty_reconstructed
            - self.obj_revenue_subsidy
        )

        # Optimal value constraint
        #
        # Since SP1 minimizes the reduced objective, SP2 must compare against the
        # same reduced value. For fixed leader decisions, this is equivalent to using
        # the original objective with the same constant added to both sides.
        follower_objective_scale = 1_000_000
        m.addConstr(
            (self.obj_total_follower_reduced / follower_objective_scale) <= (self.sp1_optimal_value / follower_objective_scale) + 1e-6,  # small tolerance to account for numerical issues
        name="OptimalityConstraint"
        )

        #endregion

    #region Objective function
    def _has_normalization_bounds(self) -> bool:
        """Check if normalization bounds are available in the instance data."""
        data = self.instance
        return all(value is not None for value in [data.Emission_min, data.Emission_max, data.Cost_min, data.Cost_max,])

    def _build_objective_components(self) -> Tuple[gp.LinExpr, gp.LinExpr]:
        """
        Build the leader objective components (emissions and costs) evaluated at the fixed leader decisions and 
        SP2 follower variables.
        """
        data = self.instance

        # 1) Transport emissions
        self.obj_emission_transport_leader = data.epsilon_truck * (
            gp.quicksum(self.q_gsw[g,s,w] * data.TD_gs[g][s] for g in data.G for s in data.S for w in data.W) +
            gp.quicksum(self.q_slw[s,l,w] * data.TD_sl[s][l] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) * data.TD_si[s][i] for s in data.S for i in data.I for w in data.W) +
            gp.quicksum(self.q_scw[s,c,w] * data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
        )

        # 2) Treatment emissions
        self.obj_emission_treatment_leader = (
            gp.quicksum(data.epsilon_land[w] * self.q_slw[s,l,w] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum(data.epsilon_inc[w] * (self.q_siw[s,i,w] + self.d_siw[s,i,w]) for s in data.S for i in data.I for w in data.W)
        )

        # 3) Fuel emissions (cement kiln)
        self.obj_emission_fuel_leader = (
            gp.quicksum(data.epsilon_kiln_f[f] * self.q_cf[c,f] for c in data.C for f in data.F) +
            gp.quicksum(data.epsilon_kiln_w[w] * self.q_scw[s,c,w] for s in data.S for c in data.C for w in data.W)
        )

        self.obj_total_env_leader = self.obj_emission_transport_leader + self.obj_emission_treatment_leader + self.obj_emission_fuel_leader

        # 4) Transport costs
        self.obj_cost_transport_leader = data.c_truck * (
            gp.quicksum(self.q_gsw[g,s,w] * data.TD_gs[g][s] for g in data.G for s in data.S for w in data.W) +
            gp.quicksum(self.q_slw[s,l,w] * data.TD_sl[s][l] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) * data.TD_si[s][i] for s in data.S for i in data.I for w in data.W)
        )

        # 5) Treatment costs
        self.obj_cost_treatment_leader = (
            data.c_land * gp.quicksum(self.q_slw[s,l,w] for s in data.S for l in data.L for w in data.W) +
            data.c_inc * gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) for s in data.S for i in data.I for w in data.W)
        )

        # 6) Subsidy cost
        self.obj_cost_subsidy_leader = gp.quicksum(
            self.q_scw[s,c,w] * gp.quicksum(data.phi_wh[w][h]*self.z_wh[w,h] for h in data.H) for s in data.S for c in data.C for w in data.W
        )

        self.obj_total_mon_leader = self.obj_cost_transport_leader + self.obj_cost_treatment_leader + self.obj_cost_subsidy_leader

        return self.obj_total_env_leader, self.obj_total_mon_leader

    def get_leader_objective_components(self) -> dict[str, float]:
        """Evaluate leader objective components at the SP2 follower-optimal solution."""
        if self.model is None:
            raise RuntimeError("Model is not built yet. Call build() before getting objective components."
            )

        if self.model.SolCount == 0:
            raise RuntimeError("SP2 has no solution. Cannot evaluate leader objective components."
            )

        return {
            "Transport emissions": float(self.obj_emission_transport_leader.getValue()),
            "Treatment emissions": float(self.obj_emission_treatment_leader.getValue()),
            "Fueling emissions": float(self.obj_emission_fuel_leader.getValue()),
            "Total emissions (raw)": float(self.obj_total_env_leader.getValue()),
            f"Total emissions objective term (weighted {self.instance.weight_env:.2f}, normalized if bounds exist)": float(self.obj_total_env_weighted_leader.getValue()),

            "Transport cost": float(self.obj_cost_transport_leader.getValue()),
            "Treatment cost": float(self.obj_cost_treatment_leader.getValue()),
            "Subsidy cost": float(self.obj_cost_subsidy_leader.getValue()),
            "Total costs (raw)": float(self.obj_total_mon_leader.getValue()),
            f"Total costs objective term (weighted {self.instance.weight_mon:.3f}, normalized if bounds exist)": float(self.obj_total_mon_weighted_leader.getValue()),

            "Objective value (raw)": float(self.obj_total_env_leader.getValue() + self.obj_total_mon_leader.getValue()),
            "Objective value (weighted sum)": float(self.model.ObjVal),
        }


    def _set_objective(self, *, objective_scale: float = 1.0) -> None:
        '''
        Set objective function for SP2 equal to leader objectiv, using the same weighted normalization as the master problem
        (can be zero for plain feasibility check, but for Yue logic equals leader objective
        to allow dual information for optimality cut generation)
        '''
        data = self.instance
        m = self.model

        E, C = self._build_objective_components()

        if self._has_normalization_bounds():
            E_range = data.Emission_max - data.Emission_min
            C_range = data.Cost_max - data.Cost_min

            if E_range <= 1e-6:
                logging.warning("Normalization range for emissions is very small (Emission_max - Emission_min <= 1e-6). Emission component is set to 0.")
                E_normalized = 0
            else:
                E_normalized = (E - data.Emission_min) / E_range

            if C_range <= 1e-6:
                logging.warning("Normalization range for costs is very small (Cost_max - Cost_min <= 1e-6). Cost component is set to 0.")
                C_normalized = 0
            else:
                C_normalized = (C - data.Cost_min) / C_range

            # Optional: scale the objective to get values in a more reasonable range for Gurobi (e.g., between 1 and 100) to help with numerical stability 
            # and solver performance; does not change the optimal solution or the shape of the Pareto front, just scales the objective values
            # objective_scale = 100

            self.obj_total_env_weighted_leader = objective_scale * (data.weight_env * E_normalized)
            self.obj_total_mon_weighted_leader = objective_scale * (data.weight_mon * C_normalized)

        else:
            self.obj_total_env_weighted_leader = data.weight_env * E
            self.obj_total_mon_weighted_leader = data.weight_mon * C

        objective = self.obj_total_env_weighted_leader + self.obj_total_mon_weighted_leader

        m.setObjective(objective, GRB.MINIMIZE)
    #endregion