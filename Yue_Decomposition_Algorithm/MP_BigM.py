import gurobipy as gp
from gurobipy import GRB
import logging
import math

from Instances.instance_generator import InstanceData

from typing import Dict, Tuple, Any
from dataclasses import dataclass


# Class for storing Master Problem solution
@dataclass
class MasterSolution:
    """Data class to store Master Problem solution"""
    mp_obj: float
    mp_bound: float                             # Store MP bound (e.g., for gap analysis)
    q_gsw: Dict[Tuple[int, int, int], float]    # Waste flow from generation g to transfer s
    q_slw: Dict[Tuple[int, int, int], float]    # Waste flow from transfer s to landfill l
    q_siw: Dict[Tuple[int, int, int], float]    # Waste flow from transfer s to incinerator i
    d_siw: Dict[Tuple[int, int, int], float]    # Residue waste flow from transfer s to incinerator i
    mu_land: float                              # Waste capacity quota for landfills
    mu_inc: float                               # Waste capacity quota for incinerators
    mu_kiln: float                              # Waste capacity quota for cement kilns
    z_wh: Dict[Tuple[int, int], int]            # Subsidy level choice for waste type w
    y_wh: Dict[Tuple[int, int], float]          # Linearization variable for subsidy cost (y_wh[w,h] = z_wh[w,h] * sum_{s,c} q_scw0[s,c,w])
    objective_components: dict[str, float] | None    # Optional dictionary to hold the components of the leader objective function for posterior analysis (e.g., transport emissions, treatment emissions, subsidy cost, etc.)

# Class for storing the KKT optimality cut model components, i.e. all variables and constraints related to the optimality cut for one l in L (for readability and debugging purposes)
@dataclass
class KKTOCBlock:
    """Data class to store variables and constraints related to one KKT optimality cut block for a specific l in L"""
    l: int  # iteration index (1,2,...)
    x_ck_fixed: Dict[Tuple[int, int], int]  # discrete follower pattern for this cut from SP1 (SP2 infeasible) or SP2 (SP2 feasible)

    # primal follower continuous vars for this block
    q_cf: gp.tupledict
    q_scw: gp.tupledict
    y_wh_KKT: gp.tupledict

    # dual vars
    # duals for inequalities (with complementarity binaries)
    lam_F3: gp.tupledict
    lam_F4: gp.tupledict
    lam_F5: gp.tupledict
    lam_F6: gp.tupledict

    # duals for non-negativity constraints of follower variables (with complementarity binaries)
    pi_q_cf: gp.tupledict
    pi_q_scw: gp.tupledict

    # complementarity binaries (one per inequality)
    bin_F3: gp.tupledict
    bin_F4: gp.tupledict
    bin_F5: gp.tupledict
    bin_F6: gp.tupledict
    bin_q_cf: gp.tupledict
    bin_q_scw: gp.tupledict

    # optional: store constraint handles for debugging
    constr: Dict[str, Any]

# Class for Master Problem (P1 with limited combinations of follower variables)
class MasterProblem:
    """
    Master Problem (Leader Problem - Municipality)
    
    The leader (municipality) determines:
    - Waste quotas "µ" for landfills, incinerators and cement facilities
    - Waste routing decisions "q" (to landfills, incinerators, cement facilities)
    - Subsidy "phi" levels for waste at each generation spot and transfer station
    
    Decision variables:
    - q_gsw: Continuous variable indicating waste "w" flow from generation "g" to transfer station "s"
    - q_slw: Continuous variable indicating waste "w" flow from transfer station "s" to landfill "l"
    - q_siw: Continuous variable indicating waste "w" flow from transfer station "s" to incinerator "i"
    - d_siw: Continuous variable indicating residue waste "w" flow from transfer station "s" to incinerator "i" (due to declined waste from cement kilns that must be incinerated instead)
    - µ_land: Continuous variable indicating waste capacity quota allocated to landfills
    - µ_inc: Continuous variable indicating waste capacity quota allocated to incinerators
    - µ_kiln: Continuous variable indicating waste capacity quota allocated to cement kilns
    - z_wh: Binary variable for choice of discrete subsidy level "h" for waste type "w"
    - y_wh: Continuous variable for linearization of subsidy cost (y_wh[w,h] = z_wh[w,h] * sum_{s,c} q_scw0[s,c,w])
    - theta_stern: Auxiliary variable for profit function approximation (cut generation)    -> not necessary to define within MP (= solution of MP and no decision variable), but can be helpful for readability and debugging (instead of using a dictionary with keys like "theta_stern_0", "theta_stern_1" etc. for multiple cuts)

    - q_scw: Flow from transfer station "s" to cement facility "c" for waste type "w" (decided in lower level!)
    """
    #region __init__ method
    # Two options for __init__:
    # (1) Object exists before the model is built 
        # - init stores data and prepares containers
        # - model is built and configured via build() method
    # (2) Setup model direcctly when initializing an object of the class
        # in this case, an object represents a model; model is always alive
        # - init creates model object
        # - build() only populates it, i.e. adds variables, constraints, objective, updates it
    def __init__(self, instance: InstanceData):
        self.instance = instance
        self.model = None
        self._build = False

        # Data containers for KKT optimality cut blocks
        self.kkt_oc_blocks: Dict[int, KKTOCBlock] = {}  # Dictionary to store KKT optimality cut blocks by iteration index l
        self._kkt_oc_counter = 0  # Counter to assign unique indices to KKT optimality cut blocks

        # Variable Containers (filled in build())
        # Leader
        self.q_gsw = None
        self.q_slw = None
        self.q_siw = None
        self.d_siw = None
        self.mu_land = None
        self.mu_inc = None
        self.mu_kiln = None
        self.z_wh = None
        self.y_wh = None

        # Dummy Follower variables
        self.x_ck0 = None
        self.q_cf0 = None
        self.q_scw0 = None

        # Store objective components for posteriori analysis
        self.obj_emission_transport = None
        self.obj_emission_treatment = None
        self.obj_emission_fuel = None
        self.obj_cost_transport = None
        self.obj_cost_treatment = None
        self.obj_cost_subsidy = None
        self.obj_total_env = None
        self.obj_total_env_weighted = None
        self.obj_total_mon = None
        self.obj_total_mon_weighted = None

        self.objective_expression = None
        self.objective_lb_cutoff = None
        self.objective_ub_cutoff = None
    #endregion

    # =============================================================================
    ######## public methods to build, solve, extract solution ########
    # =============================================================================

    #region Build model
    # everything after '*' are keyword-only arguments, i.e. must be specified by name when calling
    # 'output_flag: 1' to show Gurobi output, 0 to suppress
    # '-> None' indicates that this method does not return any value (optional hint); also '-> int | None', or '-> dict', '-> MasterSolution' etc.
    def build(self, *, name: str = "MasterProblem", output_flag: int = 1, objective_scale: float = 1.0) -> None:
        # prevent rebuilding or building a model twice (=silent bug)
        if self._build: 
            raise RuntimeError("Masterproblem model was already built.")
        
        """Build the Master Problem model"""
        self.model = gp.Model(name)
        self.model.setParam('OutputFlag', output_flag)
        
        self._add_variables()
        self._add_constraints()
        self._set_objective(objective_scale=objective_scale)
        self.model.update()
        
        logging.info("\nMaster Problem model structure (build):\n")
        logging.info(f"  → Total created variables: {self.model.NumVars}")
        logging.info(f"  → Thereof binary variables: {self.model.NumBinVars}")
        logging.info(f"  → Thereof continuous variables: {self.model.NumVars - self.model.NumBinVars}")
        logging.info(f"  → Total created constraints: {self.model.NumConstrs}\n\n")
        
        self._build = True
        # if the model shall be rebuild, set object._build = False before calling build() again
    #endregion


    #region Solve Master Problem
    def solve(self, *, time_limit: int = GRB.INFINITY, mip_gap: float = 1e-4) -> None:
        # Not necessary to check, if the model is built directly within __init__
        assert self.model is not None, "Model is not built yet. Call build() before solve()."
        
        """Solve the Master Problem"""
        logging.info("\n" + "-"*60)
        logging.info("Solving Master Problem...")
        logging.info(f"  → Time limit: {time_limit} seconds")
        logging.info("-"*60)
        logging.info("\nMaster Problem model structure:\n")
        logging.info(f"  → Total variables: {self.model.NumVars}")
        logging.info(f"  → Thereof binary variables: {self.model.NumBinVars}")
        logging.info(f"  → Thereof continuous variables: {self.model.NumVars - self.model.NumBinVars}\n")

        logging.info(f"  → Total constraints: {self.model.NumConstrs}\n\n")
        self.model.Params.TimeLimit = time_limit
        self.model.Params.Threads = 20
        self.model.Params.MIPGap = mip_gap  # Optional: set MIP gap for faster solves (e.g., 5% gap)
        self.model.Params.ScaleFlag = 2     # Enable geometric scaling to help with numerical issues and potentially improve bounds (https://link.springer.com/article/10.1007/s10589-011-9420-4)
        self.model.Params.Presolve = 2      # Enable presolve to reduce problem size and potentially improve solve times

        self.model.Params.NumericFocus = 1  # Degree to which the code attempts to detect and manage numerical issues (0 - default, 3 max)
        self.model.Params.IntFeasTol = 1e-5     # Default is 1e-5, can be tightened to 1e-6 for more precise integer solutions (at the cost of longer solve times)
        self.model.Params.IntegralityFocus = 1  # Try to avoid solutions that exploit integrality tolerances, i.e. "trickle flow"
        self.model.Params.FeasibilityTol = 1e-6  # Default is 1e-6, can be tightened to 1e-7 for more precise feasibility checks (at the cost of longer solve times)
        self.model.optimize()
    #endregion

    #region Get objective
    def get_objective_breakdown(self) -> Dict[str, float]:
        """Evaluate objective components at current incumbent."""
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError("No MP solution available.")

        return {
            "Transport emissions": float(self.obj_emission_transport.getValue()),
            "Treatment emissions": float(self.obj_emission_treatment.getValue()),
            "Fueling emissions": float(self.obj_emission_fuel.getValue()),
            "Total emissions (raw)": float(self.obj_total_env.getValue()),
            f"Total emissions objective term (weighted {self.instance.weight_env:.2f}, normalized if bounds exist)": float(self.obj_total_env_weighted.getValue()),
            "Transport cost": float(self.obj_cost_transport.getValue()),
            "Treatment cost": float(self.obj_cost_treatment.getValue()),
            "Subsidy cost": float(self.obj_cost_subsidy.getValue()),
            "Total costs (raw)": float(self.obj_total_mon.getValue()),
            f"Total costs objective term (weighted {self.instance.weight_mon:.3f}, normalized if bounds exist)": float(self.obj_total_mon_weighted.getValue()),
            "Objective value (raw)": float(self.obj_total_env.getValue() + self.obj_total_mon.getValue()),
            "Objective value (weighted sum)": float(self.model.ObjVal),
        }
    #endregion

    #region Get solution
    def extract_solution(self) -> MasterSolution:
        """Extract solution from the Master Problem"""
        if self.model.status == GRB.OPTIMAL:
            logging.info('✓ Master Problem solved optimally.')
        elif self.model.status == GRB.SUBOPTIMAL:
            logging.info('⚠ Master Problem solved suboptimally.')
        elif self.model.status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            logging.info('⚠ Master Problem solve time limit reached. Best solution found will be extracted.')
        else:
            raise RuntimeError("Master Problem is not solvable within time limit; cannot extract solution because no solution is available.")

        data = self.instance
        lead_obj_components = self.get_objective_breakdown()  # Get objective components for posterior analysis

        return MasterSolution(
            mp_obj = self.model.ObjVal,
            mp_bound = self.model.ObjBound,
            q_gsw = {(g, s, w): self.q_gsw[g, s, w].X for g in data.G for s in data.S for w in data.W},
            q_slw = {(s, l, w): self.q_slw[s, l, w].X for s in data.S for l in data.L for w in data.W},
            q_siw = {(s, i, w): self.q_siw[s, i, w].X for s in data.S for i in data.I for w in data.W},
            d_siw = {(s, i, w): self.d_siw[s, i, w].X for s in data.S for i in data.I for w in data.W},
            mu_land = self.mu_land.X,
            mu_inc = self.mu_inc.X,
            mu_kiln = self.mu_kiln.X,
            # rounding because of floating-point relaxation within gurobi (0.9999997 or 1.0000002 possible)
            z_wh = {(w, h): int(round(self.z_wh[w, h].X)) for w in data.W for h in data.H},
            y_wh = {(w, h): self.y_wh[w, h].X for w in data.W for h in data.H},
            objective_components = lead_obj_components
        )
    #endregion

    # =============================================================================
    ############ internal methods to add variables, constraints, objective ############
    # =============================================================================
    
    #region Add variables
    def _add_variables(self) -> None:
        data = self.instance
        m = self.model

        # ===== LEADER VARIABLES =====
        self.q_gsw = m.addVars(data.G, data.S, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="q_gsw")
        self.q_slw = m.addVars(data.S, data.L, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="q_slw")
        self.q_siw = m.addVars(data.S, data.I, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="q_siw")
        self.d_siw = m.addVars(data.S, data.I, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="d_siw")
        self.mu_land = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="mu_land")
        self.mu_inc = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="mu_inc")
        self.mu_kiln = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="mu_kiln")
        self.z_wh = m.addVars(data.W, data.H, vtype=GRB.BINARY, name="z_wh")
        self.y_wh = m.addVars(data.W, data.H, lb=0.0, vtype=GRB.CONTINUOUS, name="y_wh")

        # ===== DUMMY FOLLOWER VARIABLES (FOR CUT GENERATION) =====
        self.x_ck0 = m.addVars(data.C, data.K, vtype=GRB.BINARY, name="x_ck0")
        self.q_cf0 = m.addVars(data.C, data.F, lb=0.0, vtype=GRB.CONTINUOUS, name="q_cf0")
        self.q_scw0 = m.addVars(data.S, data.C, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name="q_scw0")

    #endregion

    #region Add constraints
    def _add_constraints(self) -> None:
        data = self.instance
        m = self.model

        # ==================================================== 
        # Leader constraints 
        # ====================================================
        # region Leader Constraints
        # (L1) Waste balance: all generated waste must go to some transfer station
        m.addConstrs(
            (gp.quicksum(self.q_gsw[g,s,w] for s in data.S) == data.Q_gw[g][w] for g in data.G for w in data.W),
        name="L1_wasteBalance"
        )

        # (L2) Waste dispatching: waste at a transfer station must be sent to landfill or incinerator
        # (cement kiln usage is handled by the follower, so not included here)
                # superfluous due to F7-0 !
        # m.addConstrs(
        #     (gp.quicksum(self.q_gsw[g,s,w] for g in data.G) >=
        #     gp.quicksum(self.q_slw[s,l,w] for l in data.L) + gp.quicksum(self.q_siw[s,i,w] for i in data.I)
        #     for s in data.S for w in data.W),
        # name="L2_wasteDispatching"
        # )

        # (L3) Transfer station capacity
        m.addConstrs(
            (gp.quicksum(self.q_gsw[g,s,w] for g in data.G for w in data.W) <= data.Q_s[s] for s in data.S),
        name="L3_transferCapacity"
        )

        # (L4) Landfill capacity
        m.addConstrs(
            (gp.quicksum(self.q_slw[s,l,w] for s in data.S for w in data.W) <= data.Q_l[l] for l in data.L),
        name="L4_landfillCapacity"
        )

        # Coupling constraint:
        # (L5) Residue waste routing euqals declined waste from cement kilns that must be incinerated instead (coupling constraint between landfill/incinerator routing and cement kiln decisions)
        # In the reduced model, r_sw0 is not an independent variable but an afin linear expression A_sw - sum_c q_scw0[s,c,w]
        m.addConstrs(
            (gp.quicksum(self.d_siw[s,i,w] for i in data.I) 
             == gp.quicksum(self.q_gsw[g,s,w] for g in data.G)
              - gp.quicksum(self.q_slw[s,l,w] for l in data.L)
              - gp.quicksum(self.q_siw[s,i,w] for i in data.I)
              - gp.quicksum(self.q_scw0[s,c,w] for c in data.C)  # coupling with cement kiln decisions in follower (via dummy variable q_scw0)
             for s in data.S for w in data.W),
        name="L5_residueWasteRouting"
        )

        # (L6) Incinerator capacity
        m.addConstrs(
            (gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) for s in data.S for w in data.W) <= data.Q_i[i] for i in data.I),
        name="L6_incineratorCapacity"
        )

        # (L7) Landfill transport must fulfill landfill quota
        m.addConstr(
            gp.quicksum(self.q_slw[s,l,w] for s in data.S for l in data.L for w in data.W) == self.mu_land * data.Q_gen_total,
        name="L7_landfillTransportQuota"
        )

        # (L8) Incinerator transport must fulfill incineration quota
        m.addConstr(
            gp.quicksum(self.q_siw[s,i,w] for s in data.S for i in data.I for w in data.W) == self.mu_inc * data.Q_gen_total,
        name="L8_incineratorTransportQuota"
        )

        # (L9) For each waste type, exactly one subsidy level is chosen
        m.addConstrs(
            (gp.quicksum(self.z_wh[w,h] for h in data.H) == 1 for w in data.W),
        name="L9_subsidyLevelChoice"
        )

        # Coupling constraint:
        # (L10) Total subsidy cost cannot exceed municipality budget
        # scale for better numerics o matrix an RHS, but keep feasible set untouched (e.g., if costs are in the order of 10,000 and budget is 1,000,000, scale down costs by factor of 1000 to get values in the order of 10 and keep budget at 1000)
        budget_mun_scale = 10_000
        m.addConstr(
            gp.quicksum((data.phi_wh[w][h] / budget_mun_scale) * self.y_wh[w,h] for w in data.W for h in data.H) <= data.budget_municipality / budget_mun_scale,
        name="L10_municipalityBudget"
        )
        
        # (L11) Quotas must sum to 1
        m.addConstr(
            self.mu_land + self.mu_inc + self.mu_kiln == 1,
        name="L11_quotaBalance"
        )

        # (L12) Landfill quota cannot exceed policy limit
        m.addConstr(
            self.mu_land <= data.kappa_land,
        name="L12_landfillQuotaLimit"
        )

        # Linearization constraints for subsidy cost (y_wh[w,h] = z_wh[w,h] * sum_{s,c} q_scw0[s,c,w]) via McCormick envelopes
        # (L13)
        m.addConstrs(
            (self.y_wh[w,h] <= self.z_wh[w,h] * data.U_w[w] for w in data.W for h in data.H),
        name="L13_subsidyCostLinearization_1"
        )

        # (L14)
        m.addConstrs(
            (self.y_wh[w,h] <= gp.quicksum(self.q_scw0[s,c,w] for s in data.S for c in data.C) for w in data.W for h in data.H),
        name="L14_subsidyCostLinearization_2"
        )

        # (L15)
        m.addConstrs(
            (self.y_wh[w,h] >= gp.quicksum(self.q_scw0[s,c,w] for s in data.S for c in data.C) - (1 - self.z_wh[w,h])*data.U_w[w] for w in data.W for h in data.H),
        name="L15_subsidyCostLinearization_3"
        )
        #endregion

        # ====================================================
        # Follower constraints
        # ====================================================
        # region Follower constraints 
        # (dummy vars, for cut generation)

        # (F1-0) # Only one pre- and co-processing capacity per cement facility feasible
        m.addConstrs(
            (gp.quicksum(self.x_ck0[c,k] for k in data.K) <= 1 for c in data.C),
        name="F1-0_capacityChoice"
        )

        # (F2-0) Cement facility budget constraint for investing in pre- & co-processing
        # scale numerics for better matrix and RHS vaulues, but keep feasible set untouched (e.g., if costs are in the order of 10,000 and budget is 1,000,000, scale down costs by factor of 1000 to get values in the order of 10 and keep budget at 1000)
        budget_cem_scale = 1_000_000
        m.addConstr(
            gp.quicksum(self.x_ck0[c,k] * (data.c_invest_k[k] / budget_cem_scale) for c in data.C for k in data.K) <= data.budget_cem / budget_cem_scale,
        name="F2-0_cementBudget"
        )

        # (F3-0) Energy fulfillment in cement kiln
        m.addConstrs(
            (gp.quicksum(self.q_cf0[c,f] * data.beta_f[f] for f in data.F) + gp.quicksum(self.q_scw0[s,c,w] * data.beta_w[w] for s in data.S for w in data.W) >= data.alpha_c[c] for c in data.C),
        name="F3-0_energyFulfillment"
        )

        # (F4-0) Co-processing capacity limitation
        m.addConstrs(
            (gp.quicksum(self.q_scw0[s,c,w] * data.beta_w[w] for s in data.S for w in data.W) <= data.kappa_coproc * data.alpha_c[c] for c in data.C),
        name="F4-0_coprocCapacityLimit"
        )

        # (F5-0) Pre- & co-processing capacity according to investment decision
        m.addConstrs(
            (gp.quicksum(self.q_scw0[s,c,w] for s in data.S for w in data.W) <= gp.quicksum(self.x_ck0[c,k] * data.Q_k[k] for k in data.K) for c in data.C),
        name="F5-0_investmentCapacity"
        )

        # (F6-0) Waste quota fulfillment at cement facility
        # Station-wise availability after equality elimination:
        # sum_c q_scw0[s,c,w] <= A_sw = sum_g q_gsw[g,s,w] - sum_l q_slw[s,l,w] - sum_i q_siw[s,i,w]
        # This preserves the local transfer-station balance while eliminating r_sw0
        m.addConstrs(
            (gp.quicksum(self.q_scw0[s,c,w] for c in data.C) 
             <= gp.quicksum(self.q_gsw[g,s,w] for g in data.G) 
              - gp.quicksum(self.q_slw[s,l,w] for l in data.L) 
              - gp.quicksum(self.q_siw[s,i,w] for i in data.I)
              for s in data.S for w in data.W),
        name="F6-0_stationAvailability",
        )
        #endregion
    #endregion
    
    
    #region Set objective
    def _has_normalization_bounds(self) -> bool:
        """Check if normalization bounds are available in the instance data."""
        data = self.instance
        return all(value is not None for value in [data.Emission_min, data.Emission_max, data.Cost_min, data.Cost_max])
    
    def _build_leader_objective_components(self) -> tuple[gp.LinExpr, gp.LinExpr]:
        """Build the separate components of the leader objective function for better readability and posterior analysis."""
        data = self.instance

        # ===== LEADER OBJECTIVE =====
        # 1) Transport emissions
        self.obj_emission_transport = data.epsilon_truck * (
            gp.quicksum(self.q_gsw[g,s,w] * data.TD_gs[g][s] for g in data.G for s in data.S for w in data.W) +
            gp.quicksum(self.q_slw[s,l,w] * data.TD_sl[s][l] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) * data.TD_si[s][i] for s in data.S for i in data.I for w in data.W) +
            gp.quicksum(self.q_scw0[s,c,w] * data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
        )

        # 2) Treatment emissions
        self.obj_emission_treatment = (
            gp.quicksum(data.epsilon_land[w] * self.q_slw[s,l,w] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum(data.epsilon_inc[w] * (self.q_siw[s,i,w] + self.d_siw[s,i,w]) for s in data.S for i in data.I for w in data.W)
        )

        # 3) Fuel emissions (cement kiln)
        self.obj_emission_fuel = (
            gp.quicksum(data.epsilon_kiln_f[f] * self.q_cf0[c,f] for c in data.C for f in data.F) +
            gp.quicksum(data.epsilon_kiln_w[w] * self.q_scw0[s,c,w] for s in data.S for c in data.C for w in data.W)
        )

        # 4) Transport costs
        self.obj_cost_transport = data.c_truck * (
            gp.quicksum(self.q_gsw[g,s,w] * data.TD_gs[g][s] for g in data.G for s in data.S for w in data.W) +
            gp.quicksum(self.q_slw[s,l,w] * data.TD_sl[s][l] for s in data.S for l in data.L for w in data.W) +
            gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) * data.TD_si[s][i] for s in data.S for i in data.I for w in data.W)
        )

        # 5) Treatment costs
        self.obj_cost_treatment = (
            data.c_land * gp.quicksum(self.q_slw[s,l,w] for s in data.S for l in data.L for w in data.W) +
            data.c_inc * gp.quicksum((self.q_siw[s,i,w] + self.d_siw[s,i,w]) for s in data.S for i in data.I for w in data.W)
            # (data.c_inc-data.c_penalty) * gp.quicksum(self.d_siw[s,i,w] for s in data.S for i in data.I for w in data.W)
            # (data.c_inc-1) * gp.quicksum(self.r_sw0[s,w] for s in data.S for w in data.W)
        )

        # 6) Subsidy cost
        self.obj_cost_subsidy = gp.quicksum(data.phi_wh[w][h] * self.y_wh[w,h] for w in data.W for h in data.H)
        
        # Total emissions
        self.obj_total_env = (self.obj_emission_transport + self.obj_emission_treatment + self.obj_emission_fuel)

        # Total costs
        self.obj_total_mon = (self.obj_cost_transport + self.obj_cost_treatment + self.obj_cost_subsidy)

        return self.obj_total_env, self.obj_total_mon
    
    def _set_objective(self, *, objective_scale: float = 1.0) -> None:
        data = self.instance
        m = self.model

        E, C = self._build_leader_objective_components()

        if self._has_normalization_bounds():
            E_range = data.Emission_max - data.Emission_min
            C_range = data.Cost_max - data.Cost_min

            # Check for small normalization ranges to avoid numerical issues; if range is too small, 
            # skip normalization for that component, log a warning and use fixed value 
            # - 0 (drop objective beause it does not affect the pareto dominance) or 
            # - 0.5 (neutral midpoint, such that objective contributes equally for all solutions)
            if E_range <= 1e-6:
                logging.warning("Normalization range for emissions is very small (Emission_max - Emission_min <= 1e-6). Normalization will be skipped for the emission component to avoid numerical issues.")
                E_normalized = 0 # 0.5
            else:
                E_normalized = (E - data.Emission_min) / E_range
            
            if C_range <= 1e-6:
                logging.warning("Normalization range for costs is very small (Cost_max - Cost_min <= 1e-6). Normalization will be skipped for the cost component to avoid numerical issues.")
                C_normalized = 0 # 0.5
            else:
                C_normalized = (C - data.Cost_min) / C_range

            # Optional: scale the objective to get values in a more reasonable range for Gurobi (e.g., between 1 and 100) to help with numerical stability 
            # and solver performance; does not change the optimal solution or the shape of the Pareto front, just scales the objective values
            # objective_scale = 100

            self.obj_total_env_weighted = objective_scale * (data.weight_env * E_normalized)
            self.obj_total_mon_weighted = objective_scale * (data.weight_mon * C_normalized)

        else:
            self.obj_total_env_weighted = data.weight_env * E
            self.obj_total_mon_weighted = data.weight_mon * C

        self.objective_expression = self.obj_total_env_weighted + self.obj_total_mon_weighted

        m.setObjective(self.objective_expression, GRB.MINIMIZE)

    def update_objective_cutoffs(
            self,
            *, 
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            tolerance: float = 1e-5
    ) -> None:
        
        cutoff_row_scale = 1_000        # scale for better numerics of the constraint matrix and RHS, but keep feasible set untouched
        objective_constant = self.objective_expression.getConstant()

        if lower_bound is not None and math.isfinite(lower_bound):
            rhs_lb = lower_bound - tolerance - objective_constant

            if self.objective_lb_cutoff is None:
                self.objective_lb_cutoff = self.model.addConstr(cutoff_row_scale * self.objective_expression >= cutoff_row_scale * (lower_bound - tolerance), name="Objective_Cutoff_LB")
            else:
                self.objective_lb_cutoff.RHS = cutoff_row_scale * rhs_lb

        if upper_bound is not None and math.isfinite(upper_bound):
            rhs_ub = upper_bound + tolerance - objective_constant

            if self.objective_ub_cutoff is None:
                self.objective_ub_cutoff = self.model.addConstr(cutoff_row_scale * self.objective_expression <= cutoff_row_scale * (upper_bound + tolerance), name="Objective_Cutoff_UB")
            else:
                self.objective_ub_cutoff.RHS = cutoff_row_scale * rhs_ub

        self.model.update()
    #endregion

    def _availability_expr(self, s: int, w: int) -> gp.LinExpr:
        """Return A_sw = inflow to station minus direct landfill/incineration flows."""
        data = self.instance
        return (
            gp.quicksum(self.q_gsw[g, s, w] for g in data.G)
            - gp.quicksum(self.q_slw[s, l, w] for l in data.L)
            - gp.quicksum(self.q_siw[s, i, w] for i in data.I)
        )

    # region Add KKT-OC block 
    # (after solving SP2)
    def _add_kkt_oc_block_bigM(self, x_ck_fixed: Dict[Tuple[int, int], int]) -> int:
        """
        Method to add one KKT optimality cut block for iteration l with fixed follower pattern x_ck_fixed (from SP1 or SP2)
        - creates variables and constraints for the KKT optimality cut block
        - stores them in a KKTOCBlock dataclass for readability and debugging

        Compared with the original formulation, r_sw is eliminated as an independent
        lower-level variable and reconstructed through the leader-induced availability
        A_sw. Consequently, the equality duals nu_F6 and nu_F7 as well as pi_r_sw
        disappear. A new nonnegative multiplier lam_F6[s,w] is introduced for the
        station-wise availability inequality.
        """
        data = self.instance
        m = self.model

        self._kkt_oc_counter += 1
        l = self._kkt_oc_counter
        pfx = f"OC{l}"  # prefix for variable and constraint names for this cut block

        #region Variables KKT-OC block
        # 1) Primal follower continuous variables for this cut block (same as in SP, but with suffix for this cut)
        q_cf = m.addVars(data.C, data.F, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_q_cf")
        q_scw = m.addVars(data.S, data.C, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_q_scw")
        y_wh_KKT = m.addVars(data.W, data.H, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_y_wh_KKT")  # for linearization of subsidy cost in stationarity conditions

        # 2) Dual variables for this cut block
        # duals for inequalities
        lam_F3 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_lam_F3")
        lam_F4 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_lam_F4")
        lam_F5 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_lam_F5")
        lam_F6 = m.addVars(data.S, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_lam_F6")
        # duals for non-negativity constraints of follower variables
        pi_q_cf = m.addVars(data.C, data.F, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_pi_q_cf")
        pi_q_scw = m.addVars(data.S, data.C, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_pi_q_scw")

        # 3) Complementarity binaries for this cut block
        bin_F3 = m.addVars(data.C, vtype=GRB.BINARY, name=f"{pfx}_bin_F3")
        bin_F4 = m.addVars(data.C, vtype=GRB.BINARY, name=f"{pfx}_bin_F4")
        bin_F5 = m.addVars(data.C, vtype=GRB.BINARY, name=f"{pfx}_bin_F5")
        bin_F6 = m.addVars(data.S, data.W, vtype=GRB.BINARY, name=f"{pfx}_bin_F6")
        bin_q_cf = m.addVars(data.C, data.F, vtype=GRB.BINARY, name=f"{pfx}_bin_q_cf")
        bin_q_scw = m.addVars(data.S, data.C, data.W, vtype=GRB.BINARY, name=f"{pfx}_bin_q_scw")
        #endregion

        #region Constraints KKT-OC block

        # ======================================================
        # Stationarity constraints
        # ======================================================

        # (S1) Stationarity for q_cf[c,f]
        m.addConstrs(
            (data.price_f[f] - lam_F3[c]*data.beta_f[f] - pi_q_cf[c,f] == 0 for c in data.C for f in data.F),
            name=f"{pfx}_S1_stationarity_q_cf",
        )

        # (S2) Stationarity for q_scw[s,c,w]
        m.addConstrs(
            (data.c_preproc_w[w] 
             + data.c_truck*data.TD_sc[s][c] 
             - data.c_penalty 
             - gp.quicksum(data.phi_wh[w][h]*self.z_wh[w,h] for h in data.H) 
             - lam_F3[c]*data.beta_w[w] 
             + lam_F4[c]*data.beta_w[w] 
             + lam_F5[c] 
             + lam_F6[s,w]
             - pi_q_scw[s,c,w] == 0 for s in data.S for c in data.C for w in data.W),
            name=f"{pfx}_S2_stationarity_q_scw",
        )


        # ======================================================
        # Primal feasibility constraints (same as SP, but with suffix for this cut)
        # ======================================================

        # (F1) and (F2) can be skipped because they only involve binary variables x_ck which are fixed in this cut block and do not affect the duals
        
        # (F3) Energy fulfillment in cement kiln
        # (pf1) F3 energy requirement: alpha_c - sum_f q_cf*beta_f - sum_w q_scw*beta_w <= 0
        m.addConstrs(
            (data.alpha_c[c] - gp.quicksum(q_cf[c,f]*data.beta_f[f] for f in data.F) 
             - gp.quicksum(q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) <= 0 for c in data.C),
            name=f"{pfx}_pf1_F3",
        )

        # (pf2) F4 co-processing share: sum_w q_scw*beta_w - kappa*alpha_c <= 0
        m.addConstrs(
            (gp.quicksum(q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) - data.kappa_coproc * data.alpha_c[c] <= 0 for c in data.C),
            name=f"{pfx}_pf2_F4",
        )

        # (pf3) F5 capacity with fixed x_ck pattern: sum_w q_scw - sum_k x_ck_fixed*Q_k <= 0
        m.addConstrs(
            (gp.quicksum(q_scw[s,c,w] for s in data.S for w in data.W) - gp.quicksum(x_ck_fixed[(c, k)]*data.Q_k[k] for k in data.K) <= 0 for c in data.C),
            name=f"{pfx}_pf3_F5",
        )

        # (pf4) F6 station-wise availability: sum_c q_scw - sum_g q_gsw + sum_l q_slw + sum_i q_siw <= 0
        m.addConstrs(
            (gp.quicksum(q_scw[s,c,w] for c in data.C) 
             - gp.quicksum(self.q_gsw[g, s, w] for g in data.G)
             + gp.quicksum(self.q_slw[s, l, w] for l in data.L)
             + gp.quicksum(self.q_siw[s, i, w] for i in data.I) <= 0
             for s in data.S for w in data.W),
            name=f"{pfx}_pf4_F6",
        )
        # non-negativity of primal variables is already defined in variable creation, so no need to add explicitly here

        # ======================================================
        # Complementary slackness constraints (using big-M and binaries)
        # ======================================================
        # (cs1) lam_F3[c] * b_F3[c] = 0, with b_F3[c] = (sum_f q_cf*beta_f + sum_w q_cw*beta_w - alpha_c) >= 0
        m.addConstrs(
            (lam_F3[c] <= data.M_dual["lam_F3"] * bin_F3[c] for c in data.C),
            name=f"{pfx}_CS1_dual",
        )
        m.addConstrs(
            (
                gp.quicksum(q_cf[c, f] * data.beta_f[f] for f in data.F)
                + gp.quicksum(q_scw[s, c, w] * data.beta_w[w] for s in data.S for w in data.W)
                - data.alpha_c[c]
                <= data.M_primal["F3"] * (1 - bin_F3[c])
                for c in data.C
            ),
            name=f"{pfx}_CS1_constr",
        )

        # (cs2) lam_F4[c] * b_F4[c] = 0, with b_F4[c] = (kappa*alpha_c - sum_w q_cw*beta_w) >= 0
        m.addConstrs(
            (lam_F4[c] <= data.M_dual["lam_F4"] * bin_F4[c] for c in data.C),
            name=f"{pfx}_CS2_dual",
        )
        m.addConstrs(
            (
                data.kappa_coproc * data.alpha_c[c]
                - gp.quicksum(q_scw[s, c, w] * data.beta_w[w] for s in data.S for w in data.W)
                <= data.M_primal["F4"][c] * (1 - bin_F4[c])
                for c in data.C
            ),
            name=f"{pfx}_CS2_constr",
        )

        # (cs3) lam_F5[c] * b_F5[c] = 0, with b_F5[c] = (sum_k x_ck_fixed*Q_k - sum_w q_cw) >= 0
        m.addConstrs(
            (lam_F5[c] <= data.M_dual["lam_F5"] * bin_F5[c] for c in data.C),
            name=f"{pfx}_CS3_dual",
        )
        # Use pattern-specific capacity instead of global Big-M to tighten the formulation (cap_c could be eliminated when multiplying parantheses on RHS, but kept for clarity)
        # Safety margin 1e-6 not needed beause q_scw>=0. Thus, when cap_c=0, the complementarity constraint becomes -sum_w q_cw <= 0, which is always true. The hard zero-transport condition is already imposed by primal feasibility F5.
        # When cap_c>0, the constraint is not restrictive for the primal variables when bin_F5[c]=0, and forces sum_w q_cw to be 0 when bin_F5[c]=1, thus satisfying complementarity.
        for c in data.C:
            cap_c = gp.quicksum(x_ck_fixed[(c, k)] * data.Q_k[k] for k in data.K)
            m.addConstr(
                (
                    cap_c - gp.quicksum(q_scw[s, c, w] for s in data.S for w in data.W)
                    <= cap_c * (1 - bin_F5[c])
                ),
                name=f"{pfx}_CS3_constr_c{c}",
            )
        
        # (cs4) lam_F6[s,w] * b_F6[s,w] = 0, with b_F6[s,w] = (A_sw - sum_c q_scw) >= 0
        m.addConstrs(
            (lam_F6[s, w] <= data.M_dual["lam_F6"] * bin_F6[s, w] for s in data.S for w in data.W),
            name=f"{pfx}_CS4_dual",
        )
        m.addConstrs(
            (
                gp.quicksum(self.q_gsw[g, s, w] for g in data.G)
                - gp.quicksum(self.q_slw[s, l, w] for l in data.L)
                - gp.quicksum(self.q_siw[s, i, w] for i in data.I)
                - gp.quicksum(q_scw[s, c, w] for c in data.C)
                <= data.M_primal["F6"][s][w] * (1 - bin_F6[s, w])
                for s in data.S for w in data.W
            ),
            name=f"{pfx}_CS4_constr",
        )

        # ---------------------------------------------------------------------
        # Bound complementarity (cs5)-(cs6): pi * q = 0 with pi>=0, q>=0
        # Pattern:
        #   pi <= M_pi * z
        #   q  <= M_q  * (1 - z)
        # ---------------------------------------------------------------------

        # (cs5) pi_q_scw[s,c,w] * q_scw[s,c,w] = 0
        m.addConstrs(
            (pi_q_scw[s, c, w] <= data.M_dual["pi_q_scw"] * bin_q_scw[s, c, w] for s in data.S for c in data.C for w in data.W),
            name=f"{pfx}_CS5_dual_q_scw",
        )
        m.addConstrs(
            (q_scw[s, c, w] <= data.M_primal["q_scw"][s][c][w] * (1 - bin_q_scw[s, c, w]) for s in data.S for c in data.C for w in data.W),
            name=f"{pfx}_CS5_primal_q_scw",
        )

        # (cs6) pi_q_cf[c,f] * q_cf[c,f] = 0
        m.addConstrs(
            (pi_q_cf[c, f] <= data.M_dual["pi_q_cf"] * bin_q_cf[c, f] for c in data.C for f in data.F),
            name=f"{pfx}_CS6_dual_q_cf",
        )
        m.addConstrs(
            # (q_cf[c, f] <= data.M_primal["q_cf"] * (1 - bin_q_cf[c, f]) for c in data.C for f in data.F),
            (q_cf[c, f] <= data.M_primal["q_cf"][c][f] * (1 - bin_q_cf[c, f]) for c in data.C for f in data.F),
            name=f"{pfx}_CS6_primal_q_cf",
        )

        # ======================================================
        # Yue Optimality Cut (minimization follower)
        # f(dummy vars) <= f(KKT block vars)
        # ======================================================

        # ----- LEFT HAND SIDE (dummy follower vars in MP) -----

        lhs = (
            # Coal cost
            gp.quicksum(self.q_cf0[c, f]*data.price_f[f] for c in data.C for f in data.F)
            # Investment cost
            + gp.quicksum(data.fixcost_invest_k[k]*self.x_ck0[c, k] for c in data.C for k in data.K)
            # Preprocessing
            + gp.quicksum(self.q_scw0[s, c, w] * data.c_preproc_w[w] for s in data.S for c in data.C for w in data.W)
            # Transport
            + gp.quicksum(self.q_scw0[s, c, w] * data.c_truck* data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
            # Penalty
            - gp.quicksum(self.q_scw0[s, c, w] * data.c_penalty for s in data.S for c in data.C for w in data.W)
            # Subsidy revenue (subtract)
            - gp.quicksum(self.y_wh[w, h] * data.phi_wh[w][h] for w in data.W for h in data.H)
        )

        # ----- RIGHT HAND SIDE (KKT block variables) -----

        # Linearization of KKT-block subsidy revenue on RHS
        # y_wh_KKT[w,h] = z_wh[w,h] * sum_{s,c} q_scw[s,c,w]
        m.addConstrs(
            (y_wh_KKT[w,h] <= self.z_wh[w,h]*data.U_w[w] for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def1"
        )
        m.addConstrs(
            (y_wh_KKT[w,h] <= gp.quicksum(q_scw[s,c,w] for s in data.S for c in data.C) for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def2"
        )
        m.addConstrs(
            (y_wh_KKT[w,h] >= gp.quicksum(q_scw[s,c,w] for s in data.S for c in data.C) - (1 - self.z_wh[w,h])*data.U_w[w] for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def3"
        )

        rhs = (
            # Coal cost
            gp.quicksum(q_cf[c, f] * data.price_f[f] for c in data.C for f in data.F)
            # Investment cost (fixed discrete pattern)
            + gp.quicksum(data.fixcost_invest_k[k] * x_ck_fixed[(c, k)] for c in data.C for k in data.K)
            # Preprocessing
            + gp.quicksum(q_scw[s, c, w] * data.c_preproc_w[w] for s in data.S for c in data.C for w in data.W)
            # Transport
            + gp.quicksum(q_scw[s, c, w] * data.c_truck * data.TD_sc[s][c] for s in data.S for c in data.C for w in data.W)
            # Penalty
            - gp.quicksum(q_scw[s, c, w] * data.c_penalty for s in data.S for c in data.C for w in data.W)
            # Subsidy revenue (subtract)
            - gp.quicksum(y_wh_KKT[w, h] * data.phi_wh[w][h] for w in data.W for h in data.H)
        )

        # ----- Add Optimality Cut -----
        follower_objective_scale = 1_000_000
        m.addConstr((lhs/follower_objective_scale) <= (rhs/follower_objective_scale) + 1e-6, name=f"{pfx}_OptimalityCut")     # add small tolerance to avoid numerical issues
        #endregion

        # create new KKTOCBlock with unique index l and given fixed follower pattern
        kkt_oc_block = KKTOCBlock(
            l=l,
            x_ck_fixed=x_ck_fixed,
            q_cf=q_cf, y_wh_KKT=y_wh_KKT, q_scw=q_scw,
            lam_F3=lam_F3, lam_F4=lam_F4, lam_F5=lam_F5, lam_F6=lam_F6,
            pi_q_cf=pi_q_cf, pi_q_scw=pi_q_scw,
            bin_F3=bin_F3, bin_F4=bin_F4, bin_F5=bin_F5, bin_F6=bin_F6,
            bin_q_cf=bin_q_cf, bin_q_scw=bin_q_scw,
            constr={}
        )
        self.kkt_oc_blocks[kkt_oc_block.l] = kkt_oc_block

        m.update()
        return l
    #endregion