import mujoco
import numpy as np

from dynamics import calculate_frame_dynamics, setup_robot_dynamics_parameters, transfer_iparams
from sensors.sensors import get_sensor_measurement_idx
from simulators.simulator import _refresh_derived_data


def test_clean_regressor_gt_matches_wrench_after_step() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="arm">
              <joint name="hinge" type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
              <body name="target/object">
                <inertial pos="0.1 0.02 0.03" mass="2" diaginertia="0.02 0.03 0.04"/>
                <site name="target/ft_sensor" euler="0 0 180"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor joint="hinge"/></actuator>
          <sensor>
            <force name="force" site="target/ft_sensor"/>
            <torque name="torque" site="target/ft_sensor"/>
          </sensor>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    params = setup_robot_dynamics_parameters(model, data, ee_body_name="arm")
    pose_x_sensor = params.poses.get_x_("site", "target/ft_sensor")
    pose_x_object = params.poses.get_x_("body", "target/object")
    pose_sensor_object = pose_x_sensor.inv().dot(pose_x_object)
    # Parallel-axis theorem: I_origin = I_com + m * (|com|^2 * Id - com com^T).
    gt_object = np.array([2, 0.2, 0.04, 0.06, 0.0226, 0.0518, 0.0608, -0.004, -0.0012, -0.006])
    gt_sensor = transfer_iparams(pose_sensor_object, gt_object)

    data.qpos[:] = 0.8
    data.qvel[:] = 5.0
    data.ctrl[:] = 0.2
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
