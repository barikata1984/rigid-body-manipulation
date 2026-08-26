from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from dynamics import calculate_frame_dynamics, setup_robot_dynamics_parameters, transfer_iparams
from sensors.sensors import get_sensor_measurement_idx
from simulators.setup import generate_model_data
from simulators.simulator import _refresh_derived_data


def test_clean_regressor_gt_matches_wrench_after_step() -> None:
    repo_root = Path(__file__).parents[1]
    cfg = SimpleNamespace(
        object=repo_root / "xml_models/targets/loaded_dice",
        manipulator=repo_root / "xml_models/manipulators/sequential",
        reset_keyframe="initial_state",
        recorder=SimpleNamespace(track_cam_distance_factor=5.0, track_cam_name="tracking"),
    )
    model, data, gt = generate_model_data(cfg)
    mujoco.mj_forward(model, data)

    params = setup_robot_dynamics_parameters(model, data)
    pose_x_sensor = params.poses.get_x_("site", "target/ft_sensor")
    pose_x_object = params.poses.get_x_("body", "target/object")
    pose_sensor_object = pose_x_sensor.inv().dot(pose_x_object)
    gt_object = np.array([gt["mass"], *(gt["mass"] * gt["com"]), *gt["globalinertia"]])
    gt_sensor = transfer_iparams(pose_sensor_object, gt_object)

    data.qpos[:] = [0.05, -0.04, 0.03, 0.8, -0.6, 0.4]
    data.qvel[:] = [0.2, -0.1, 0.3, 5.0, -4.0, 3.0]
    data.ctrl[:] = [1.0, -0.5, 0.25, 0.2, -0.3, 0.4]
    mujoco.mj_step(model, data)
    _refresh_derived_data(model, data)

    jointvars = np.stack((data.qpos.copy(), data.qvel.copy(), data.qacc.copy()))
    _, _, regressor = calculate_frame_dynamics(
        jointvars,
        params.inverse_dynamics,
        params.id_ll,
        params.poses.x_b[params.id_ll],
        params.pose_ll_llj,
        pose_x_sensor,
    )
    force_idx = get_sensor_measurement_idx(model, name="force")
    torque_idx = get_sensor_measurement_idx(model, name="torque")
    wrench = np.concatenate((data.sensordata[force_idx], data.sensordata[torque_idx]))

    np.testing.assert_allclose(regressor @ gt_sensor, wrench, rtol=1e-13, atol=1e-14)
