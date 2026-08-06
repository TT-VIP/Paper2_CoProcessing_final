from pathlib import Path
import json
from typing import Any, Dict, List

from Instances.archive_scripts.instance_generator import InstanceData 

#region JSON reader
# JSON turns dict keys into strings, so it is necessary to convert them back to ints for indexed Big-M values
def _keys_to_int(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}

def _keys_to_int_recursive(obj):
    if isinstance(obj, dict):
        return {int(k): _keys_to_int_recursive(v) for k, v in obj.items()}
    return obj

# JSON to InstanceData object
def read_instance_data_from_json(json_path: Path) -> InstanceData:
    with json_path.open('r', encoding='utf-8') as file:
        payload = json.load(file)

    metadata = payload["metadata"]
    data = payload["data"]

    # Convert lists back to ranges for index sets
    data['G'] = range(data['G_max'])
    data['S'] = range(data['S_max'])
    data['W'] = range(data['W_max'])
    data['I'] = range(data['I_max'])
    data['L'] = range(data['L_max'])
    data['C'] = range(data['C_max'])
    data['K'] = range(data['K_max'])
    data['F'] = range(data['F_max'])
    data['H'] = range(data['H_max'])

    # Convert JSON string keys back to ints for coordinates (Dict[int, Tuple[float, float]])
    data['G_coords'] = {int(k): tuple(v) for k, v in data['G_coords'].items()}
    data['S_coords'] = {int(k): tuple(v) for k, v in data['S_coords'].items()}
    data['I_coords'] = {int(k): tuple(v) for k, v in data['I_coords'].items()}
    data['L_coords'] = {int(k): tuple(v) for k, v in data['L_coords'].items()}
    data['C_coords'] = {int(k): tuple(v) for k, v in data['C_coords'].items()}

    # Convert JSON string keys back to ints for indexed Big-M values
    data['M_primal']['F4'] = _keys_to_int(data['M_primal']['F4'])
    data['M_primal']['F6'] = _keys_to_int_recursive(data['M_primal']['F6'])
    data['M_primal']['r_sw'] = _keys_to_int_recursive(data['M_primal']['r_sw'])
    data['M_primal']['q_cf'] = _keys_to_int_recursive(data['M_primal']['q_cf'])
    data['M_primal']['q_scw'] = _keys_to_int_recursive(data['M_primal']['q_scw'])

    data["instance_name"] = metadata["instance_id"]
    data["instance_size_class"] = metadata["size_class"]
    data["instance_regime"] = metadata["structural_regime"]

    return InstanceData(**data)
    # instance = InstanceData(
    #     G_max=data['G_max'], S_max=data['S_max'], W_max=data['W_max'], 
    #     I_max=data['I_max'], L_max=data['L_max'], C_max=data['C_max'],
    #     K_max=data['K_max'], F_max=data['F_max'], H_max=data['H_max'],

    #     G=data['G'], S=data['S'], W=data['W'], I=data['I'], L=data['L'], 
    #     C=data['C'], K=data['K'], F=data['F'], H=data['H'],

    #     G_coords=data['G_coords'], S_coords=data['S_coords'], I_coords=data['I_coords'],
    #     L_coords=data['L_coords'], C_coords=data['C_coords'],
        
    #     cement_names=data['cement_names'],
        
    #     TD_gs=data['TD_gs'], TD_sl=data['TD_sl'], 
    #     TD_sc=data['TD_sc'], TD_si=data['TD_si'], 
    #     TD_si_avg=data['TD_si_avg'],
        
    #     epsilon_truck=data['epsilon_truck'],
    #     epsilon_land=data['epsilon_land'], epsilon_inc=data['epsilon_inc'], 
    #     epsilon_kiln_w=data['epsilon_kiln_w'], epsilon_kiln_f=data['epsilon_kiln_f'],

    #     c_truck=data['c_truck'], c_land=data['c_land'], c_inc=data['c_inc'],

    #     Q_gw=data['Q_gw'], Q_gen_total=data['Q_gen_total'],
    #     Q_s=data['Q_s'], Q_l=data['Q_l'], Q_i=data['Q_i'],
    #     Q_k=data['Q_k'], Q_k_max=data['Q_k_max'],

    #     weight_env=data['weight_env'], weight_mon=data['weight_mon'],

    #     kappa_land=data['kappa_land'], kappa_coproc=data['kappa_coproc'],

    #     budget_municipality=data['budget_municipality'],
    #     budget_cem=data['budget_cem'],

    #     phi_max=data['phi_max'],
    #     phi_wh=data['phi_wh'],

    #     price_f=data['price_f'], 
    #     beta_f=data['beta_f'], beta_w=data['beta_w'],

    #     alpha_c=data['alpha_c'], 

    #     c_invest_k=data['c_invest_k'], 
    #     c_preproc_w=data['c_preproc_w'],
    #     c_penalty=data['c_penalty'], 

    #     tau = data["tau"],
    #     fixcost_invest_k = data["fixcost_invest_k"],

    #     M_primal=data['M_primal'], M_dual=data['M_dual'],
    #     U_w=data['U_w']
    # )

    # return instance
#endregion

def read_instance_metadata_from_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open('r', encoding='utf-8') as file:
        payload = json.load(file)
    return payload["metadata"]