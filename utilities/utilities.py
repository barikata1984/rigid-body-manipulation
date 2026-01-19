import math
from collections.abc import Iterable
from types import SimpleNamespace

from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id
from omegaconf import OmegaConf


def register_omegaconf_resolvers():
    """Register custom OmegaConf resolvers for math expressions in YAML.
    
    Usage in YAML:
        ${pi:}        -> 3.14159...
        ${eval:2*pi}  -> 6.28318...
        ${eval:pi/2}  -> 1.5707...
    """
    OmegaConf.register_new_resolver("pi", lambda: math.pi, replace=True)
    OmegaConf.register_new_resolver(
        "eval",
        lambda expr: eval(expr, {"pi": math.pi, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}),
        replace=True
    )


# Auto-register resolvers when this module is imported
register_omegaconf_resolvers()


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
