from collections.abc import Iterable

import numpy as np
from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id


def classify_dict_kargs(dict_kargs):
    arr_like = {}
    others = {}

    for k, v in dict_kargs.items():
        if isinstance(v, str):
            others[k] = v
        elif isinstance(v, Iterable):
            arr_like[k] = v
        else:
            others[k] = v

    return arr_like, others


def get_element_id(m, elem_type, name):
    obj_enum = None

    if "body"== elem_type:
        obj_enum = mjtObj.mjOBJ_BODY
    elif "camera"== elem_type:
        obj_enum = mjtObj.mjOBJ_CAMERA
    elif "joint"== elem_type:
        obj_enum = mjtObj.mjOBJ_JOINT
    elif "sensor"== elem_type:
        obj_enum = mjtObj.mjOBJ_SENSOR
    elif "site"== elem_type:
        obj_enum = mjtObj.mjOBJ_SITE
    elif "keyframe"== elem_type:
        obj_enum = mjtObj.mjOBJ_KEY
    elif "numeric"== elem_type:
        obj_enum = mjtObj.mjOBJ_NUMERIC
    else:
        raise ValueError(f"'{elem_type}' is not supported for now. Use mj_name2id and check the value of an ID instead.")

    id = mj_name2id(m, obj_enum, name)

    if -1 == id:
        raise ValueError(f"ID for '{name}' not found. Check the manipulator .xml or the object .xml")

    return id


def split_iterables(frames, train_ratio=0.7, seed=0):
    """
    Splits the indices of a list into training and testing sets.

    Args:
        data_list: The list of dictionaries.
        train_ratio: The proportion of data to use for training (default 0.7).

    Returns:
        A tuple containing two lists: train_indices and test_indices.
    """
    n = len(frames)
    num_train = int(n * train_ratio)

    # Get shuffled indices for the whole dataset
    rng = random.default_rng(seed)
    all_indices = list(range(n))
    rng.shuffle(all_indices)

    train_frames = frames[all_indices[:num_train]]
    test_frames = frames[all_indices[num_train:]]

    return dict(train=train_frames, test=test_frames)

