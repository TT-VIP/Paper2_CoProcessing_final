import gurobipy as gp
from gurobipy import GRB
import logging
import math

from Instances.instance_generator import InstanceData

from typing import Dict, Tuple, Any, Optional
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

    # primal slack variables for SOS1 (one per inequality)
    s_F3: gp.tupledict
    s_F4: gp.tupledict
    s_F5: gp.tupledict
    s_F6: gp.tupledict

    # optional: store constraint handles for debugging
    constr: Dict[str, Any]

    # store bounds determined in OC-block
    U_Q_w: Dict[int, float]
    U_sw_F6_l: Optional[Dict[Tuple[int, int], float]] = None
    

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


    #region Solve the MP
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
#        self.model.printStats()             # Print model statistics (number of variables, constraints, nonzeros, etc.) before solving for better understanding of model size and complexity
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap  # Optional: set MIP gap for faster solves (e.g., 5% gap)
        self.model.Params.ScaleFlag = 2     # Enable geometric scaling to help with numerical issues and potentially improve bounds (https://link.springer.com/article/10.1007/s10589-011-9420-4)
        self.model.Params.Presolve = 2      # Enable presolve to reduce problem size and potentially improve solve times

        self.model.Params.NumericFocus = 1  # Degree to which the code attempts to detect and manage numerical issues (0 - default, 3 max)
        self.model.Params.IntFeasTol = 1e-5     # Default is 1e-5, can be tightened to 1e-6 for more precise integer solutions (at the cost of longer solve times)
        self.model.Params.PreSOS1BigM = 0       # Disable presolve reduction of big-M values for SOS1 constraints to prevent numerical issues
        self.model.Params.IntegralityFocus = 1  # Focus on integrality to improve solution reliability for MP, which is a mixed-integer problem
        self.model.Params.FeasibilityTol = 1e-6  # Default is 1e-6, can be tightened to 1e-7 for more precise feasibility checks (at the cost of longer solve times)
        self.model.optimize()
#        self.model.printQuality()  # Print solution quality information (e.g., MIP gap, bound, etc.) after solve
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
        
        cutoff_row_scale = 1_000  # Scale the cutoff row to avoid numerical issues with very small coefficients in the objective expression
        objective_constant = self.objective_expression.getConstant()
        
        if lower_bound is not None and math.isfinite(lower_bound):
            rhs_lb = lower_bound - tolerance - objective_constant
            
            if self.objective_lb_cutoff is None:
                # During creation, Gurobi automatically moves the obj constant to the RHS
                self.objective_lb_cutoff = self.model.addConstr(cutoff_row_scale * self.objective_expression >= cutoff_row_scale * (lower_bound - tolerance), name="Objective_Cutoff_LB")
            else:
                # When updating RHS, the constant mus be included explicitly because the stored row is constant free
                # (Gurobi automatically moves the constant to the RHS during creation, so it misses in lhs)
                self.objective_lb_cutoff.RHS = cutoff_row_scale * rhs_lb

        if upper_bound is not None and math.isfinite(upper_bound):
            rhs_ub = upper_bound + tolerance - objective_constant

            if self.objective_ub_cutoff is None:
                self.objective_ub_cutoff = self.model.addConstr(cutoff_row_scale * self.objective_expression <= cutoff_row_scale * (upper_bound + tolerance), name="Objective_Cutoff_UB")
            else:
                self.objective_ub_cutoff.RHS = cutoff_row_scale * rhs_ub

        self.model.update()
    #endregion

    #region Compute LP-Bounds
    def _availability_expr(self, s: int, w: int) -> gp.LinExpr:
        """Return A_sw = inflow to station minus direct landfill/incineration flows."""
        data = self.instance
        return (
            gp.quicksum(self.q_gsw[g, s, w] for g in data.G)
            - gp.quicksum(self.q_slw[s, l, w] for l in data.L)
            - gp.quicksum(self.q_siw[s, i, w] for i in data.I)
        )
    
    def compute_availability_bounds_lp(
        self,
        *,
        time_limit_per_lp: float = 10.0,
        feasibility_tol: float = 1e-6,
        output_flag: int = 0,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute LP-based upper bounds for

            A_sw = sum_g q_gsw[g,s,w] - sum_l q_slw[s,l,w] - sum_i q_siw[s,i,w]

        by maximizing A_sw over the continuous relaxation of the current base MP.

        Intended use:
            call once directly after mp.build(...) and before adding SOS1 KKT blocks.

        Validity:
            If called before KKT-OC blocks are added, the LP relaxation contains all later
            MP feasible leader solutions. Therefore, the resulting bounds remain valid
            for all later KKT blocks.
        """
        assert self.model is not None, "Model must be built before computing bounds."

        data = self.instance
        self.model.update()

        # Structural fallback bound
        def qgen_w(w: int) -> float:
            return float(sum(data.Q_gw[g][w] for g in data.G))

        # Min of generated waste and transfer station capacity is valid upper bound for A_sw, i.e.
        # the maximum amount of waste available at a station s for co-processing is limited by the total amount of waste generated of type w and the transfer station capacity.
        structural_U_A = {
            (s, w): min(qgen_w(w), float(data.Q_s[s]))
            for s in data.S for w in data.W
        }

        # Work on a relaxed copy, not on the original MP
        lp = self.model.relax()
        lp.Params.OutputFlag = output_flag
        lp.Params.TimeLimit = time_limit_per_lp
        lp.Params.FeasibilityTol = feasibility_tol

        # Optional: repeated objective changes often benefit from dual simplex
        lp.Params.Method = 1

        U_A_sw = {}

        for s in data.S:
            for w in data.W:
                expr = (
                    gp.quicksum(lp.getVarByName(f"q_gsw[{g},{s},{w}]") for g in data.G)
                    - gp.quicksum(lp.getVarByName(f"q_slw[{s},{l},{w}]") for l in data.L)
                    - gp.quicksum(lp.getVarByName(f"q_siw[{s},{i},{w}]") for i in data.I)
                )

                lp.setObjective(expr, GRB.MAXIMIZE)
                lp.optimize()

                fallback = structural_U_A[(s, w)]

                if lp.Status == GRB.OPTIMAL:
                    bound = float(lp.ObjVal)

                    # Numerical safety and structural cap
                    bound = max(0.0, bound)
                    bound = min(bound, fallback)
                    # bound = min(bound + feasibility_tol, fallback)
                    

                    U_A_sw[(s, w)] = bound

                elif lp.Status in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and lp.SolCount > 0:
                    # Still use only if a valid upper bound is available
                    # For maximization, ObjBound is an upper bound
                    bound = float(lp.ObjBound)
                    bound = max(0.0, bound)
                    bound = min(bound, fallback)
                    # bound = min(bound + feasibility_tol, fallback)
                    U_A_sw[(s, w)] = bound

                else:
                    logging.warning(
                        f"[U_A_LP] Could not compute LP bound for (s={s}, w={w}). "
                        f"Using structural fallback {fallback:.6f}."
                    )
                    U_A_sw[(s, w)] = fallback

                logging.info(
                    f"[U_A_LP] (s={s}, w={w}): "
                    f"LP bound = {U_A_sw[(s, w)]:.6f}, "
                    f"structural bound = {fallback:.6f}, "
                    f"improvement = {fallback - U_A_sw[(s, w)]:.6f}"
                )

        self.U_A_sw_lp = U_A_sw
        return U_A_sw
    #endregion

    # region Add KKT-OC block 
    # (after solving SP2)
    def _add_kkt_oc_block_sos1(self, x_ck_fixed: Dict[Tuple[int, int], int], *, primal_dual_streghtening: bool = True) -> int:
        """
        Method to add one KKT optimality cut block for iteration l with fixed follower pattern x_ck_fixed (from SP1 or SP2)
        => using SOS1 constraints for complementarity instead of Big-M and binaries. Key idea:
            for each complementarity pair (dual >= 0) ⟂ (slack >= 0), impose SOS1 constraint on (dual, slack)
            to enforce that at most one of them can be positive, thus enforcing complementarity without big-M.
        - creates variables and constraints for the KKT optimality cut block
        - stores them in a KKTOCBlock dataclass for readability and debugging

        block uses:
        - reduced lower-level primal feasibility with r_sw eliminated,
        - stationarity and nonnegative dual feasibility,
        - SOS1 complementarity pairs instead of big-M constraints,
        - the Yue/You optimality cut, and
        - the Kleinert-type primal-dual strengthening inequality based on weak duality combined with a 
        closed-form pattern-specific upper bound 
        """
        data = self.instance
        m = self.model

        self._kkt_oc_counter += 1
        l = self._kkt_oc_counter
        pfx = f"OC{l}"  # prefix for variable and constraint names for this cut block

        # ------------------------------------------------------------------
        # Helpers and structural bounds for this fixed investment pattern
        # ------------------------------------------------------------------
        def _qgen_w(w: int) -> float:
            return float(sum(data.Q_gw[g][w] for g in data.G))

        def _cap_c(c: int) -> float:
            return float(sum(int(round(x_ck_fixed[(c, k)])) * data.Q_k[k] for k in data.K))

        cap_c = {c: _cap_c(c) for c in data.C}

        # U_A_sw >= A_sw := sum_g q_gsw - sum_l q_slw - sum_i q_siw
        # Conservative but valid: A_sw <= min(total generation of type w, transfer capacity of station s)
        if self.U_A_sw_lp is not None:
            U_A_sw = dict(self.U_A_sw_lp)
        else:
            U_A_sw = {
                (s, w): min(_qgen_w(w), float(data.Q_s[s]))
                for s in data.S for w in data.W
            }

        #region Bound A_sw
        def _build_active_f6_bounds(
                cap_c: Dict[int, float],
                U_A_sw_globalLP: Dict[Tuple[int, int], float],
        ) -> Dict[Tuple[int, int], float]:
            """
            Bound A_sw on the branch where lambda_F6[s,w] can be positive
            
            Exploit SOS1 structure:
            If lambda_F6[s,w] > 0, then s_F6[s,w] = 0 (i.e. it is active), 
            because the SOS1 constraint enforces that one of (lambda_F6[s,w], s_F6[s,w]) must be zero
            -> lambda_F6[s,w] > 0 means A_sw = sum_c q_scw[s,c,w] (the station availability is fully used)

            If lambda_F6[s,w] = 0, then s_F6[s,w] >= 0, 
            which means the station availability constraint is not binding, and A_sw can be greater than sum_c q_scw[s,c,w]


            Derivation of the pattern-specific upper bound U_sw_F6_l:

            For a fixed investment pattern, the flow to each cement plant is limited by
            (i) the installed co-processing capacity (F5) and
            (ii) the maximum waste amount implied by the co-processing energy limit (F4)

            Summing these plant-wise limits over all cement plants yields a tight,
            pattern-dependent bound on the total flow from transfer station s to waste
            fraction w. The resulting bound is combined with the global LP-based bound
            (and optionally the transfer station and waste availability limits) by taking
            their minimum

            Compared to the global bound alone, this exploits the currently selected
            investment pattern and therefore provides a substantially tighter coefficient
            for the F6 dual variable and the corresponding McCormick linearization
            """
            # data = self.instance
            active_bounds: Dict[Tuple[int, int], float] = {}

            for s in data.S:
                for w in data.W:
                    pattern_acceptance_bound = sum(min(cap_c[c], data.kappa_coproc * data.alpha_c[c] / data.beta_w[w]) for c in data.C)
                    active_bounds[(s, w)] = max(0.0, min(pattern_acceptance_bound, U_A_sw_globalLP[(s, w)]))

            return active_bounds
        #endregion

        # Upper bound for Q_w^l = sum_{s,c} q_scw[s,c,w] used in y_wh = z_wh * Q_w^l.
        # Composite structural bound from generation, installed capacity, station availability,
        # and the energy-based co-processing limit.
        ##################################### OLD VERSION ###########################################
        # U_Q_w = {}
        # for w in data.W:
        #     gen_bound = _qgen_w(w)
        #     investment_bound = sum(cap_c[c] for c in data.C)
        #     availability_bound = sum(U_A_sw[(s, w)] for s in data.S)
        #     energy_bound = sum(data.kappa_coproc * data.alpha_c[c] / data.beta_w[w] for c in data.C)
        #     U_Q_w[w] = float(min(gen_bound, investment_bound, availability_bound, energy_bound))
        U_Q_w = {}
        for w in data.W:
            gen_bound = _qgen_w(w)
            availability_bound = sum(U_A_sw[(s, w)] for s in data.S)
            pattern_bound = sum(min(cap_c[c], data.kappa_coproc * data.alpha_c[c] / data.beta_w[w]) for c in data.C)
            U_Q_w[w] = float(min(gen_bound, availability_bound, pattern_bound))

        #region Variables KKT-OC block
        # ------------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------------
        # 1) Primal follower continuous variables for this cut block (same as in SP, but with suffix for this cut)
        q_cf = m.addVars(data.C, data.F, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_q_cf")
        q_scw = m.addVars(data.S, data.C, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_q_scw")
        # eta_wh = z_wh * Q_w^l, where Q_w^l = sum_{s,c} q_scw[s,c,w] 
        # is the total waste of type w co-processed in cement kilns in this cut block; used for linearization of subsidy cost in stationarity conditions
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
        s_F3 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_s_F3")
        s_F4 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_s_F4")
        s_F5 = m.addVars(data.C, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_s_F5")
        s_F6 = m.addVars(data.S, data.W, lb=0.0, vtype=GRB.CONTINUOUS, name=f"{pfx}_s_F6")
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
        
        # Gurobi requires two variables for SOS1 constraints, so we define the slack variable as slack = inequality >= 0, and the dual variable as lam >= 0, 
        # and then impose SOS1 on (lam, slack) to enforce that at most one of them can be positive, thus enforcing complementarity without big-M.

        # no direct primal feasibility constraint for inequalities needed, because slack = constraint; and slack >= 0
        # (F3) Energy fulfillment in cement kiln
        # original: sum_f beta_f q_cf + sum_w beta_w q_scw >= alpha_c
        # slack_F3[c] = energy supplied - alpha_c >= 0
        m.addConstrs(
            (s_F3[c] 
             == gp.quicksum(q_cf[c,f]*data.beta_f[f] for f in data.F) 
              + gp.quicksum(q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) 
              - data.alpha_c[c] for c in data.C),
            name=f"{pfx}_slack_pf1_F3",
        )

        # (pf2) F4 co-processing energy limit
        # original: sum_w beta_w q_scw <= kappa_coproc * alpha_c
        # slack_F4[c] = kappa_coproc * alpha_c - waste_energy >= 0
        m.addConstrs(
            (s_F4[c] == 
             data.kappa_coproc * data.alpha_c[c]
             - gp.quicksum(q_scw[s,c,w]*data.beta_w[w] for s in data.S for w in data.W) 
             for c in data.C),
            name=f"{pfx}_slack_pf2_F4",
        )

        # (pf3) F5 investment capacity 
        # original: sum_sw q_scw <= sum_k x_ck_fixed Q_k
        # slack_F5[c] = installed_capacity - used_capacity >= 0
        m.addConstrs(
            (s_F5[c] == 
            #  gp.quicksum(x_ck_fixed[(c, k)]*data.Q_k[k] for k in data.K)
             cap_c[c]
             - gp.quicksum(q_scw[s,c,w] for s in data.S for w in data.W) 
             for c in data.C),
            name=f"{pfx}_slack_pf3_F5",
        )

        # (pf4) F6 stationbalance: 
        # original: sum_c q_scw <= - sum_g q_gsw - sum_l q_slw - sum_i q_siw
        # slack_F6[s,w] = sum_g q_gsw - sum_l q_slw - sum_i q_siw - sum_c q_scw >= 0
        m.addConstrs(
            (s_F6[s, w] == 
             self._availability_expr(s,w)
             - gp.quicksum(q_scw[s, c, w] for c in data.C)
             for s in data.S for w in data.W),
            name=f"{pfx}_slack_pf4_F6",
        )
        # non-negativity of primal variables is already defined in variable creation, so no need to add explicitly here

        # ======================================================
        # Complementary slackness constraints via SOS1
        # ======================================================
        
        # Inequality constraint complementarity:
        # lam >= 0  ⟂  slack >= 0, i.e.
        #   lam_F3[c] ⟂ slack_F3[c]
        #   lam_F4[c] ⟂ slack_F4[c]
        #   lam_F5[c] ⟂ slack_F5[c]
        #  lam_F6[s,w] ⟂ slack_F6[s,w]
        # Implement as SOS1([lam, slack])
        for c in data.C:
            m.addSOS(GRB.SOS_TYPE1, [lam_F3[c], s_F3[c]], [1.0, 2.0])
            m.addSOS(GRB.SOS_TYPE1, [lam_F4[c], s_F4[c]], [1.0, 2.0])
            m.addSOS(GRB.SOS_TYPE1, [lam_F5[c], s_F5[c]], [1.0, 2.0])
        
        for s in data.S:
            for w in data.W:
                m.addSOS(GRB.SOS_TYPE1, [lam_F6[s,w], s_F6[s,w]], [1.0, 2.0])

        # Bound complementarity:
        # pi >= 0  ⟂  q >= 0, i.e.
        #   pi_q_cf[c,f] ⟂ q_cf[c,f]
        #   pi_q_scw[s,c,w] ⟂ q_scw[s,c,w]
        for c in data.C:
            for f in data.F:
                m.addSOS(GRB.SOS_TYPE1, [pi_q_cf[c, f], q_cf[c, f]], [1.0, 2.0])

        for s in data.S:
            for c in data.C:
                for w in data.W:
                    m.addSOS(GRB.SOS_TYPE1, [pi_q_scw[s, c, w], q_scw[s, c, w]], [1.0, 2.0])
        

        # Linearization of KKT-block subsidy revenue on RHS
        # ------------------------------------------------------------------
        # eta_wh = z_wh * Q_w^l, Q_w^l = sum_{s,c} q_scw[s,c,w]
        # ------------------------------------------------------------------
        # y_wh_KKT[w,h] = z_wh[w,h] * sum_{s,c} q_scw[s,c,w]
        m.addConstrs(
            (y_wh_KKT[w,h] <= self.z_wh[w,h]*U_Q_w[w] for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def1"
        )
        m.addConstrs(
            (y_wh_KKT[w,h] <= gp.quicksum(q_scw[s,c,w] for s in data.S for c in data.C) for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def2"
        )
        m.addConstrs(
            (y_wh_KKT[w,h] >= gp.quicksum(q_scw[s,c,w] for s in data.S for c in data.C) - (1 - self.z_wh[w,h])*U_Q_w[w] for w in data.W for h in data.H),
            name=f"{pfx}_y_wh_KKT_def3"
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

        m.addConstr(lhs <= rhs + 1e-6, name=f"{pfx}_OptimalityCut")     # add small tolerance to avoid numerical issues
        #endregion

        # ------------------------------------------------------------------
        # Kleinert-type primal-dual strengthening inequality for the reduced LL LP
        # Theta_tilde(q,z) >= alpha*lambda_F3 - kappa*alpha*lambda_F4
        #                    - Kbar*lambda_F5 - U_A*lambda_F6
        # Fixed investment costs are deliberately omitted on both sides.
        # ------------------------------------------------------------------
        if primal_dual_streghtening:
            theta_tilde = (
                gp.quicksum(q_cf[c, f] * data.price_f[f] for c in data.C for f in data.F)
                + gp.quicksum(
                    q_scw[s, c, w] * (data.c_preproc_w[w] + data.c_truck * data.TD_sc[s][c] - data.c_penalty)
                    for s in data.S for c in data.C for w in data.W
                    )
                - gp.quicksum(data.phi_wh[w][h] * y_wh_KKT[w, h] for w in data.W for h in data.H)
            )

            U_sw_F6_l = _build_active_f6_bounds(cap_c=cap_c, U_A_sw_globalLP=U_A_sw)
            # Log patern-speciic F6 bounds
            logging.info(f"\n{pfx} pattern-specific F6 bounds (U_sw_F6_l):")
            for (s, w) in sorted(U_sw_F6_l.keys()):
                logging.info(f"  U_sw_F6_l [s={s}, w={w}] = {U_sw_F6_l[(s, w)]:.6f}")

            dual_lower_bound = (
                gp.quicksum(data.alpha_c[c] * lam_F3[c] for c in data.C)
                - gp.quicksum(data.kappa_coproc * data.alpha_c[c] * lam_F4[c] for c in data.C)
                - gp.quicksum(cap_c[c] * lam_F5[c] for c in data.C)
                - gp.quicksum(U_sw_F6_l[(s, w)] * lam_F6[s, w] for s in data.S for w in data.W)
            )

            m.addConstr(theta_tilde >= dual_lower_bound, name=f"{pfx}_KleinertPrimalDualStrengthening")

        # create new KKTOCBlock with unique index l and given fixed follower pattern
        kkt_kwargs = dict(
            l=l,

            x_ck_fixed=x_ck_fixed,
            q_cf=q_cf,
            q_scw=q_scw,
            y_wh_KKT=y_wh_KKT,

            lam_F3=lam_F3, 
            lam_F4=lam_F4, 
            lam_F5=lam_F5, 
            lam_F6=lam_F6,
            pi_q_cf=pi_q_cf, 
            pi_q_scw=pi_q_scw,

            s_F3=s_F3,
            s_F4=s_F4,
            s_F5=s_F5,
            s_F6=s_F6,
            constr={},

            U_Q_w=U_Q_w,
        )
        
        if primal_dual_streghtening:
            kkt_kwargs["U_sw_F6_l"] = U_sw_F6_l

        kkt_oc_block = KKTOCBlock(**kkt_kwargs)
        self.kkt_oc_blocks[kkt_oc_block.l] = kkt_oc_block

        m.update()
        return l
    #endregion