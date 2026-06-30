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

# Per-joint stiffness (Nm/rad) and damping (Nm*s/rad). The iiwa14 has
# decreasing inertia from base to tip, so we taper the gains accordingly.
KP = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 50.0])
KD = np.array([ 20.0,  20.0,  15.0,  15.0,  10.0,  8.0,  5.0])


def compute_gravity(model, data):
    """Return the gravity torque vector for the current configuration.

    qfrc_bias = C(q, dq) * dq + g(q). When dq = 0 this equals g(q) exactly,
    which is what we want for gravity compensation. For the more general case
    (nonzero dq) you'd subtract Coriolis terms separately, but for a slow
    impedance task qfrc_bias is a good approximation.
    """
    return data.qfrc_bias.copy()


def impedance_torque(model, data, q_des, kp, kd):
    q = data.qpos[:7]
    dq = data.qvel[:7]
    tau = -kp * (q - q_des) - kd * dq
    tau += compute_gravity(model, data)[:7]
    return tau


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Start the arm at the desired pose so we observe how the controller holds
    # it against gravity, not a transient swing-up.
    data.qpos[:7] = Q_DES
    mujoco.mj_forward(model, data)

    print(f"Loaded {SCENE_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Kp = {KP}")
    print(f"Kd = {KD}")
    print("Launching viewer. Ctrl+drag a link to apply external force and feel the impedance.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            tau = impedance_torque(model, data, Q_DES, KP, KD)
            # Clip to actuator ctrlrange to stay within the model's declared
            # torque limits.
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.clip(tau, lo, hi)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing: sleep the remainder of model.opt.timestep.
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print(f"Viewer closed after {time.time() - sim_start:.1f}s")


if __name__ == "__main__":
    main()
