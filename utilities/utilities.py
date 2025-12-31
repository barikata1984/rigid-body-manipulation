from collections.abc import Iterable
from types import SimpleNamespace

from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id


def json_to_namespace(data):
    """Recursively convert dict to SimpleNamespace for dot-access."""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: json_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [json_to_namespace(item) for item in data]
    else:
        return data


def categorize_dict_kargs(dict_kargs):
    arr_like = {}
    others = {}

    for k, v in dict_kargs.items():
        if isinstance(v, Iterable) and not isinstance(v, str):
            arr_like[k] = v
        else:
            others[k] = v

    return arr_like, others


def get_element_id(m, elem_type, name):
    obj_enum = None

    if "body" == elem_type:
        obj_enum = mjtObj.mjOBJ_BODY
    elif "camera" == elem_type:
        obj_enum = mjtObj.mjOBJ_CAMERA
    elif "joint" == elem_type:
        obj_enum = mjtObj.mjOBJ_JOINT
    elif "sensor" == elem_type:
        obj_enum = mjtObj.mjOBJ_SENSOR
    elif "site" == elem_type:
        obj_enum = mjtObj.mjOBJ_SITE
    elif "keyframe" == elem_type:
        obj_enum = mjtObj.mjOBJ_KEY
    elif "numeric" == elem_type:
        obj_enum = mjtObj.mjOBJ_NUMERIC
    else:
        raise ValueError(
            f"'{elem_type}' is not supported for now. Use mj_name2id and check the value of an ID instead."
        )

    id = mj_name2id(m, obj_enum, name)

    if -1 == id:
        raise ValueError(f"ID for '{name}' not found. Check the manipulator .xml or the object .xml")

    return id
