import subprocess
from collections.abc import Iterable

from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id


def get_git_branch_name():
    """現在のGitブランチ名を取得します。"""
    try:
        # `git rev-parse --abbrev-ref HEAD` コマンドを実行
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        # 標準出力をトリムして返す
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Gitコマンドが失敗した場合や、Gitがインストールされていない場合
        return None


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
