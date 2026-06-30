"""Joint-space impedance control for the KUKA LBR iiwa14 in MuJoCo.

Impedance control law (joint space):
    tau = M(q) * ddq_des + h(q, dq) - Kp * (q - q_des) - Kd * (dq - dq_des)

A common simplification (used here) is the gravity-compensated PD form:
    tau = -Kp * (q - q_des) - Kd * dq + g(q)

where g(q) is the gravity torque vector. This produces a passive spring-damper
behavior around q_des while compensating for gravity, so the arm feels
compliant if you push on it (in the viewer, grab a body with ctrl+drag).

The gains Kp and Kd shape the apparent inertia, stiffness, and damping at
each joint. Lower Kp = softer / more compliant. Damping ratio ~1 with
Kd = 2*sqrt(Kp) for a unit-inertia approximation is a reasonable starting
point.

Run:
    python3 scripts/joint_impedance_control.py
"""
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer


SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "kuka_lbr", "scene.xml",
)

# Desired joint configuration (radians). A mild "ready" pose so the arm isn't
# collapsed on itself at startup. All zeros works too but is less visually
# interesting.
Q_DES = np.array([0.0, -0.6, 0.0, 1.4, 0.0, -0.8, 0.0])

# --- Joint Space Gains ---
# Per-joint stiffness (Nm/rad) and damping (Nm*s/rad).
KP_JOINT = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 50.0])
KD_JOINT = np.array([ 20.0,  20.0,  15.0,  15.0,  10.0,  8.0,  5.0])

# --- Cartesian Space Gains (from opspace.py) ---
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]
damping_ratio = 1.0

# Twist computation gains
Kpos = 0.95
Kori = 0.95
integration_dt = 1.0

# Null-space joint gains
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

# Pre-compute combined stiffness and damping arrays
KP_CART = np.concatenate([impedance_pos, impedance_ori], axis=0)
damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
KD_CART = np.concatenate([damping_pos, damping_ori], axis=0)

SITE_NAME = "attachment_site"
MOCAP_NAME = "target"


def compute_gravity(model, data):
    """Return the gravity torque vector for the current configuration."""
    return data.qfrc_bias.copy()


def joint_impedance_torque(model, data, q_des, kp, kd):
    q = data.qpos[:7]
    dq = data.qvel[:7]
    tau = -kp * (q - q_des) - kd * dq
    tau += compute_gravity(model, data)[:7]
    return tau


def cartesian_opspace_torque(model, data, site_name, mocap_name, q0):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
    mocap_id = model.body(mocap_name).mocapid[0]
    
    # Pre-allocate arrays
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    
    # Spatial velocity (aka twist)
    dx = data.mocap_pos[mocap_id] - data.site_xpos[site_id]
    twist[:3] = Kpos * dx / integration_dt
    
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt
    
    # Jacobian
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)
    jac = np.vstack((jac_p, jac_r))
    
    # Compute the task-space inertia matrix
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
        
    # Compute generalized forces for primary task
    tau = jac.T @ Mx @ (KP_CART * twist - KD_CART * (jac @ data.qvel))
    
    # Add joint task in nullspace
    Jbar = M_inv @ jac.T @ Mx
    ddq = Kp_null * (q0 - data.qpos[:7]) - Kd_null * data.qvel[:7]
    tau_null = (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq
    tau[:7] += tau_null[:7]
    
    # Add gravity compensation
    tau += data.qfrc_bias
    
    return tau[:7]


def main():
    parser = argparse.ArgumentParser(description="Impedance control for KUKA LBR")
    parser.add_argument("--mode", type=str, choices=["joint", "cartesian"], default="joint",
                        help="Control mode: 'joint' or 'cartesian'")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Start the arm at the desired pose
    data.qpos[:7] = Q_DES
    mujoco.mj_forward(model, data)
    
    # Initialize mocap target pose to match end-effector
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    mocap_id = model.body(MOCAP_NAME).mocapid[0]
    data.mocap_pos[mocap_id] = data.site_xpos[site_id].copy()
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
    data.mocap_quat[mocap_id] = site_quat.copy()

    print(f"Loaded {SCENE_PATH}")
    print(f"Control Mode: {args.mode.upper()}")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.mode == "cartesian":
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
        sim_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            if args.mode == "joint":
                tau = joint_impedance_torque(model, data, Q_DES, KP_JOINT, KD_JOINT)
            else:
                tau = cartesian_opspace_torque(model, data, SITE_NAME, MOCAP_NAME, Q_DES)
                
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.clip(tau, lo, hi)

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print(f"Viewer closed after {time.time() - sim_start:.1f}s")


if __name__ == "__main__":
    main()
