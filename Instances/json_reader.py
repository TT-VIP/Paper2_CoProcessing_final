from pathlib import Path
import json
from typing import Any, Dict

from .instance_generator import InstanceData

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
    # data['M_primal']['r_sw'] = _keys_to_int_recursive(data['M_primal']['r_sw'])
    data['M_primal']['q_cf'] = _keys_to_int_recursive(data['M_primal']['q_cf'])
    data['M_primal']['q_scw'] = _keys_to_int_recursive(data['M_primal']['q_scw'])

    data["instance_name"] = metadata["Instance Name"]
    data["instance_size_class"] = metadata["Size Class"]
    data["instance_regime"] = metadata["Structural Regime"]

    return InstanceData(**data)
#endregion

def read_instance_metadata_from_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open('r', encoding='utf-8') as file:
        payload = json.load(file)
    return payload["metadata"]