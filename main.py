import os
import cv2
import numpy as np
import mujoco as mj
import planners as pln
import matplotlib as mpl
import transformations as tf
from visualization import axes_plot_frc, ax_plot_lines_w_tgt
from matplotlib import pyplot as plt
from utilities import store
from dynamics import compose_sinert_i, inverse
from scipy import linalg as la
from math import pi, radians as rad, degrees as deg


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=3, suppress=True)


def main():
    xml_file = "obj_w_links.xml"
    xml_path = os.path.join("./xml_models", xml_file)
    print(f"Loaded xml file: {xml_file}")
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    # Setup rendering conditoins and instantiate a VideoWriter
    codec_4chr = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec_4chr)
    fps = 60  # Rendering frequency [Hz]
    height = 600
    width = 960
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    renderer = mj.Renderer(m, height, width)

    # Enable joint visualization option
    # scene_option = mj.MjvOption()
    # scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    # renderer.update_scene(data, scene_option=scene_option)

    # Prepare variables to srore dynamics data
    print("Number of")
    nv = m.nv
    print(f"    coorindates in joint space (nv): {nv:>2}")
    nu = m.nu
    print(f"    degrees of freedom (nu):         {nu:>2}")
    na = m.na
    print(f"    actuator activations (na):       {na:>2}")
    nsensordata = m.nsensordata
    print(f"    sensor outputs (nsensordata):    {nsensordata:>2}")

    # Numerically compute A and B with a finite differentiation
    epsilon = 1e-6  # Differential displacement
    centered = True  # Use the centred differentiation; False for the forward
    # Initilize state-space and cost matrices
    A = np.zeros((2 * nv + na, 2 * nv + na))  # State matrix
    B = np.zeros((2 * nv + na, nu))  # Input matrix
    C = None  # Ignore C in this code
    D = None  # Ignore D as well
    Q = np.eye(2 * nv)  # State cost matrix
#    Q[3, 3] = 1e-3*2
    R = np.eye(nu)  # Input cost matrix
    init_rot_weight = 1e+6
    for i in range(3, nu):
        R[i, i] *= init_rot_weight
#    print(f"R: {R}")

    # Compute the feedback gain matrix K
    mj.mjd_transitionFD(m, d, epsilon, centered, A, B, C, D)
    P = la.solve_discrete_are(A, B, Q, R)
    K = la.pinv(R + B.T @ P @ B) @ B.T @ P @ A
#    print(f"K: {K}")

    # Description of suffixes used from the section below:
    #   This   |        |
    #  project | MuJoCo | Description
    # ---------+--------+------------
    #    _x    |  _x    | Described in {cartesian} or {world}
    #    _b    |  _b    | Descried in {body)
    #    _i    |  _i    | Described in {principal body}
    #    _q    |  _q    | Described in the joint space
    #    _xi   |  _xi   | Of {principal} rel. to {world}
    #    _ab   |   -    | Of {body} rel. to {parent body}
    #    _ba   |   -    | Of {parent body} rel. to {body}
    #    _cb   |   -    | Of {body}  rel. to {child body}
    #    _bc   |   -    | Of {child body}  rel. to {body}
    #
    # Compose the principal spatial inertia matrix for each link including the
    # worldbody
    sinert_i = np.array([
        compose_sinert_i(m, i) for m, i in zip(m.body_mass, m.body_inertia)])
    # Convert sinert_i to sinert_b rel2 the body frame
    sinert_b = np.empty((0, 6, 6))
    for p, q, si_i in zip(m.body_ipos, m.body_iquat, sinert_i):
        Ad_SE3_ib = tf.tquat2SE3(p, q) .inv().adjoint()
        converted = Ad_SE3_ib.T @ si_i @ Ad_SE3_ib
        sinert_b = np.append(sinert_b, converted[np.newaxis, :], axis=0)
#    print(f"(sinert_(i, b).shape: ({sinert_i.shape}, {sinert_b.shape})")
#    print(f"sinert_b.shape:\n{sinert_b}")

    # Configure SE3 of child frame rel2 parent frame (M_{i, i - 1} in MR)
    SE3_home_ba = [  # ba = 00, 10, 21, ..., 65
        tf.tquat2SE3(p, q).inv() for p, q in zip(m.body_pos, m.body_quat)]
    # Configure SE3 of each body frame rel2 worldbody (M_{i} = M_{0, i} in MR)
    SE3_home_xb = [SE3_home_ba[0].inv()]  # xb = 00, 01, ..., 06
    for SE3_h_ba in SE3_home_ba[1:]:
        SE3_home_xb.append(SE3_home_xb[-1].dot(SE3_h_ba.inv()))
#    print(f"len(link_SE3_home_(ba, xb)): ({len(link_SE3_home_ba)}, {len(link_SE3_home_xb)})")
#    print(f"link_SE3_home_ba:\n{link_SE3_home_ba}")

    # Obtain unit screw axes rel2 each link = body (A_{i} in MR)
    screw_bb = np.zeros((m.body_jntnum.sum(), 6))  # bb = (11, 22, ..., 66)
    for b, (type, ax) in enumerate(zip(m.jnt_type, m.jnt_axis), 0):
        slicer = 3 * (type - 2)  # type is 2 for slide and 3 for hinge
        screw_bb[b, slicer:slicer + 3] = ax / la.norm(ax)
#    print(f"len(screw_bb)): {len(screw_bb)}")
#    print(f"screw_bb:\n{screw_bb}")

    # Ingredients for twist-wrench inverse dynamics
    # Set simulation time and timeste
    duration = 5  # Simulation time [s]
    timestep = mj.MjOption().timestep  # 0.002 [s] (500 [Hz]) by default
#    print(f"Timestep (freq.): {timestep} [s] ({1/timestep} [Hz])")
    n_steps = int(duration / timestep)
#    print(f"# of steps: {n_steps}")
    # Determin start and end displacements of the target trajectory
    start_q = np.array([0, 0, 0, 0, 0, 0])
    goal_q = np.array([0.2, 0.4, 0.6, 0.2 * pi, 0.3 * pi, 0.4 * pi])
    # Generate a trajectory planner
    plan_traj = pln.generate_5th_spline_traj_planner(
        start_q, goal_q, timestep, n_steps)

    # Ingredients for dynamics calculation
    gacc_x = np.zeros(6)
    gacc_x[:3] = mj.MjOption().gravity
    twist_00 = np.zeros(6)
    print(f"twist_00: {twist_00}")
    dtwist_00 = -gacc_x
    print(f"dtwist_00: {dtwist_00}")

    # Data storage
    traj = np.empty((0, 3, 6))
    # Cartesian coordinates of the object
    obj_pos_x = np.empty((0, 3))
    # Joint postions, velocities, and accelerations included in the model
    # expressed in the joint space
    qpos, qvel, qacc = np.empty((3, 0, nu))
    # Residual of qpos computed with mj_differentiatePos()
    res_qpos = np.empty(nu)  # residual os joint positions
    # For control signals
    tgt_ctrl, res_ctrl, ctrl = np.empty((3, 0, nu))
    # Miscoellanious
    sensordata = np.empty((0, nsensordata))
    frame_count = 0
    time = []

    for i in range(n_steps):
        tgt_traj = plan_traj(i)
        traj = store(tgt_traj, traj)
        wrench_q = inverse(
            traj[-1], SE3_home_ba, sinert_b, screw_bb, twist_00, dtwist_00)
        tgt_ctrl = store(wrench_q[:nu], tgt_ctrl)

        # Compute the feedback gain matrix K
#        mj.mjd_transitionFD(m, d, epsilon, centered, A, B, C, D)
#        P = la.solve_discrete_are(A, B, Q, R)
#        K = la.pinv(R + B.T @ P @ B) @ B.T @ P @ A

        # Retrieve the current state in q
        qpos = store(d.qpos, qpos)
        qvel = store(d.qvel, qvel)
        qacc = store(d.qacc, qacc)
        mj.mj_differentiatePos(  # Use this func to differenciate quat properly
            m,  # MjModel
            res_qpos,  # dqpos_data_buffer
            nu,  # idx of a joint up to which res_qpos are calculated
            qpos[-1],  # current qpos
            traj[-1, 0, :nu])  # target qpos or next qpos to calkculate dqvel
        res_qvel = traj[-1, 1, :nu] - qvel[-1]
        res_state = np.concatenate((res_qpos, res_qvel))
        res_ctrl = store(-K @ res_state, res_ctrl)  # Note the minus before K
        ctrl = store(tgt_ctrl[-1, :nu] + res_ctrl[-1], ctrl)
        d.ctrl = ctrl[-1]

        # Store other necessary data
        sensordata = store(d.sensordata.copy(), sensordata)
        obj_pos_x = store(d.xpos[-1], obj_pos_x)
        time.append(d.time)

        # Store frames following the fps
        if frame_count <= time[-1] * fps:
#            print(f"res_state:\n{res_state}")
            renderer.update_scene(d)
            img = renderer.render()[:, :, [2, 1, 0]]
            out.write(img)
            frame_count += 1

        # Evolute the simulation
        mj.mj_step(m, d)

    # Terminate the VideoWriter
    out.release()

    act_qpos, act_qvel, act_qfrc = np.split(
        sensordata, [1 * nu, 2 * nu], axis=1)

    # Set line attributes
    t_clip = len(time)
    time = time[:t_clip]
    # Plot the actual and target trajctories
    d_clip = min(3, nu)
    qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
    qpos_fig.suptitle("qpos")
    qpos_axes[1].set(xlabel="time [s]")
    ax_plot_lines_w_tgt(
        qpos_axes[0], time, qpos[:, :d_clip], traj[:, 0, :d_clip], "q0-2 [m]")
    if 3 < nu:
        ax_plot_lines_w_tgt(
            qpos_axes[1], time, qpos[:, 3:], traj[:, 0, 3:nu], "q3-5 [rad]")

    # Plot forces
    ctrl_fig, ctrl_axes = plt.subplots(3, 1, sharex="col", tight_layout=True)
    ctrl_fig.suptitle("act_qfrc VS tgt_ctrl")
    ctrl_axes[0].set(ylabel="q0-1 [N]")
    ctrl_axes[1].set(ylabel="q2 [N]")
    ctrl_axes[2].set(xlabel="time [s]")
    axes_plot_frc(
        ctrl_axes[:2], time, act_qfrc[:, :d_clip], tgt_ctrl[:, :d_clip])
    if 3 < nu:
        ax_plot_lines_w_tgt(
            ctrl_axes[2], time, act_qfrc[:, 3:], tgt_ctrl[:, 3:], "q3-5 [N·m]")

    plt.show()


if __name__ == "__main__":
    main()
