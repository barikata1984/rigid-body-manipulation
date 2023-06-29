import os
import cv2
import json
import numpy as np
import mujoco as mj
import dynamics as dyn
import matplotlib as mpl
import visualization as vis
import transformations as tf
# from visualization import ax_plot_lines, axes_plot_frc, ax_plot_lines_w_tgt
from matplotlib import pyplot as plt
from configure import load_configs, plan_trajectory
from utilities import store
from scipy import linalg
from math import tan, atan2, pi, radians as rad, degrees as deg


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=3, suppress=True)


def main():
    m, d, t, c = load_configs("obj_w_links.xml")

    out = cv2.VideoWriter(c.output_file_name, c.fourcc, t.fps, (c.width, c.height))
    renderer = mj.Renderer(m, c.height, c.width)

    dqpos = np.array([0.2, 0.4, 0.6, 0.2 * pi, 0.3 * pi, 0.4 * pi])
    plan_traj = plan_trajectory(m, d, t, "init_pose", dqpos)

    # Enable joint visualization option
    # scene_option = mj.MjvOption()
    # scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    # renderer.update_scene(data, scene_option=scene_option)

    # Numerically compute A and B with a finite differentiation
    epsilon = 1e-6  # Differential displacement
    centered = True  # Use the centred differentiation; False for the forward
    # Initilize state-space and cost matrices
    A = np.zeros((2 * m.nv + m.na, 2 * m.nv + m.na))  # State matrix
    B = np.zeros((2 * m.nv + m.na, m.nu))  # Input matrix
    C = None  # Ignore C in this code
    D = None  # Ignore D as well
    Q = np.eye(2 * m.nv)  # State cost matrix
#    Q[3, 3] = 1e-3*2
    R = np.eye(m.nu)  # Input cost matrix
    init_rot_weight = 1e+6
    for i in range(m.nu - 3, m.nu):
        R[i, i] *= init_rot_weight
#    print(f"R: {R}")

    # Compute the feedback gain matrix K
    mj.mjd_transitionFD(m, d, epsilon, centered, A, B, C, D)
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
#    print(f"K: {K}")

    # Description of suffixes used from the section below:
    #   This   |        |
    #  project | MuJoCo | Description
    # ---------+--------+------------
    #    _x    |  _x    | Described in {cartesian} or {world}
    #    _b    |  _b    | Descried in {body)
    #    _i    |  _i    | Described in {principal} of each body
    #    _q    |  _q    | Described in the joint space
    #    _xi   |  _xi   | Of {principal} rel. to {world}
    #    _ab   |   -    | Of {body} rel. to {parent}
    #    _ba   |   -    | Of {parent} rel. to {body}
    #    _m    |   -    | Described in the frame a joint corresponds to
    #
    # Compose the principal spatial inertia matrix for each link including the
    # worldbody
    sinert_i = np.array([
        dyn.compose_sinert_i(m, i) for m, i in zip(m.body_mass, m.body_inertia)])
    # Convert sinert_i to sinert_b rel2 the body frame
    pose_ib = tf.posquat2SE3(m.body_ipos, m.body_iquat)
    sinert_b = dyn.transfer_sinert(sinert_i, pose_ib)

    # Configure SE3 of child frame rel2 parent frame (M_{i, i - 1} in MR)
    pose_home_ba = tf.posquat2SE3(m.body_pos, m.body_quat)
    # Configure SE3 of each body frame rel2 worldbody (M_{i} = M_{0, i} in MR)
    pose_home_xb = [pose_home_ba[0].inv()]  # xb = 00, 01, ..., 06
    for p_h_ba in pose_home_ba[1:]:
        pose_home_xb.append(pose_home_xb[-1].dot(p_h_ba.inv()))

    # Obtain unit screw rel2 each link = body (A_{i} in MR)
    uscrew_bb = np.zeros((m.body_jntnum.sum(), 6))  # bb = (11, 22, ..., 66)
    for b, (type, ax) in enumerate(zip(m.jnt_type, m.jnt_axis), 0):
        slicer = 3 * (type - 2)  # type is 2 for slide and 3 for hinge
        uscrew_bb[b, slicer:slicer + 3] = ax / linalg.norm(ax)

    print("Worldbody (d)twist for inv. dyn. ==========")
    ## Set a twist vector for the worldbody
    twist_00 = np.zeros(6)
    ## Set a twist vector for the worldbody
    gacc_x = np.zeros(6)
    gacc_x[:3] = mj.MjOption().gravity
    dtwist_00 = -gacc_x  # set below to cancel out joint forces and torques due to gravity
    print(f"    twist_00:  {twist_00}")
    print(f"    dtwist_00: {dtwist_00}")

    # =*=*=*=*=*=*=*=*= Data storage =*=*=*=*=*=*=*=*=
    # trajectroy
    traj = np.empty((0, 3, 6))
    # Cartesian coordinates of the object
    obj_pos_x = np.empty((0, 3))
    # Joint variables
    qpos, qvel, qacc = np.empty((3, 0, m.nu))
    # Residual of qpos
    res_qpos = np.empty(m.nu)
    # Control signals
    tgt_ctrl, res_ctrl, ctrl = np.empty((3, 0, m.nu))
    # Dictionary to be converted to .json for training 
    model_input = {"camera_angle_x": c.cam_fovx, "frames": list()}
    # Others
    sensordata = np.empty((0, m.nsensordata))
    frame_count = 0
    time = []
    cam_xtf = []

    obs_dir = "./data/images"
    if not os.path.isdir(obs_dir):
        os.makedirs(obs_dir)

    for i in range(t.n_steps):
        tgt_traj = plan_traj(i)
        traj = store(tgt_traj, traj)
        wrench_q = dyn.inverse(
            traj[-1], pose_home_ba, sinert_b, uscrew_bb, twist_00, dtwist_00)
        tgt_ctrl = store(wrench_q[:m.nu], tgt_ctrl)

        # Retrieve joint variables
        qpos = store(d.qpos, qpos)
        qvel = store(d.qvel, qvel)
        qacc = store(d.qacc, qacc)
        # Residual of state
        mj.mj_differentiatePos(  # Use this func to differenciate quat properly
            m,  # MjModel
            res_qpos,  # data buffer for the residual of qpos 
            m.nu,  # idx of a joint up to which res_qpos are calculated
            qpos[-1],  # current qpos
            traj[-1, 0, :m.nu])  # target qpos or next qpos to calkculate dqvel
        res_state = np.concatenate((res_qpos, traj[-1, 1, :m.nu] - qvel[-1]))
        # Compute and set control, or actuator inputs
        res_ctrl = store(-K @ res_state, res_ctrl)  # Note the minus before K
        ctrl = store(tgt_ctrl[-1, :m.nu] + res_ctrl[-1], ctrl)
        d.ctrl = ctrl[-1]

        # Evolute the simulation
        mj.mj_step(m, d)

        # Store other necessary data
        sensordata = store(d.sensordata.copy(), sensordata)
        obj_pos_x = store(d.xpos[-1], obj_pos_x)
        time.append(d.time)

        # Store frames following the fps
        if frame_count <= time[-1] * t.fps:
            # Save a video and sensor measurments
            renderer.update_scene(d, c.cam_id)
            img = renderer.render()[:, :, [2, 1, 0]]
            file_path = os.path.join(obs_dir, f"{frame_count:04}")
            cv2.imwrite(os.path.join(file_path, ".png"), img)
            out.write(img)
            # Prepare ingredients for .json file
            cam_xtf = tf.trzs2SE3(d.cam_xpos[c.cam_id], d.cam_xmat[c.cam_id])
            frame = {
                "file_path": file_path, 
                "transform_matrix": cam_xtf.as_matrix().tolist(), 
                "ft": sensordata[-1, 3 * m.nu:].tolist()}

            model_input["frames"].append(frame)

            frame_count += 1

    # Terminate the VideoWriter
    out.release()

    _, _, sens_qfrc, sens_ft = np.split(
        sensordata, [1 * m.nu, 2 * m.nu, 3 * m.nu], axis=1)

    with open("./data/model_input.json", "w") as f:
        json.dump(model_input, f, indent=2)

    # =*=*=*=*=*= Plot trajectory & wrench *=*=*=*=*=*
    t_clip = len(time)
    time = time[:t_clip]
    # Plot the actual and target trajctories
    d_clip = min(3, m.nu)
    qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
    qpos_fig.suptitle("qpos")
    qpos_axes[1].set(xlabel="time [s]")
    vis.ax_plot_lines_w_tgt(
        qpos_axes[0], time, qpos[:, :d_clip], traj[:, 0, :d_clip], "q0-2 [m]")
    vis.ax_plot_lines_w_tgt(
        qpos_axes[1], time, qpos[:, 3:], traj[:, 0, 3:m.nu], "q3-5 [rad]")

    # Plot forces
    ctrl_fig, ctrl_axes = plt.subplots(3, 1, sharex="col", tight_layout=True)
    ctrl_fig.suptitle("act_qfrc VS tgt_ctrl")
    ctrl_axes[0].set(ylabel="q0-1 [N]")
    ctrl_axes[1].set(ylabel="q2 [N]")
    ctrl_axes[2].set(xlabel="time [s]")
    vis.axes_plot_frc(
        ctrl_axes[:2], time, sens_qfrc[:, :d_clip], tgt_ctrl[:, :d_clip])
    vis.ax_plot_lines_w_tgt(
        ctrl_axes[2], time, sens_qfrc[:, 3:], tgt_ctrl[:, 3:], "q3-5 [N·m]")

    # Plot ft mesurements
    ft_fig, ft_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
    ft_fig.suptitle("ft")
    vis.ax_plot_lines(
        ft_axes[0], time, sens_ft[:, :3], "x/y/z-frc of {s} [N]")
    vis.ax_plot_lines(
        ft_axes[1], time, sens_ft[:, 3:], "x/y/z-trq of {s} [N·m]")

    plt.show()


if __name__ == "__main__":
    main()
