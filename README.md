# Kuka Impedance Control

This project simulates impedance control for a KUKA LBR iiwa14 robot arm using MuJoCo. It implements two control modes — **joint-space** and **Cartesian-space** impedance — both built on a gravity-compensated control law that makes the arm behave like a compliant spring-damper. You can interact with the arm in real time by applying external forces in the viewer, and in Cartesian mode you can drag a target the end-effector will follow.

## Installation

To install the required dependencies for this project, use the provided `requirements.txt` file. It's recommended to install these into a Python virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation with the provided script. It launches a MuJoCo viewer and runs the controller in a real-time loop:

```bash
# Joint-space impedance (default)
python3 scripts/joint_impedance_control.py

# Cartesian-space impedance
python3 scripts/joint_impedance_control.py --mode cartesian
```

The `--mode` flag selects the controller:

| Mode | Behavior |
|------|----------|
| `joint` (default) | The arm holds a fixed joint configuration (`Q_DES`) like a set of springs, compensating for gravity. Push it and it springs back to the pose. |
| `cartesian` | The end-effector tracks the green target sphere (a MuJoCo *mocap* body). Move the sphere and the arm follows it; the redundant joints are resolved with a null-space task. |

### How the control law works

Both modes command **joint torques** directly (the model uses `motor` actuators, so `data.ctrl` is a torque in Nm).

- **Joint mode** uses a gravity-compensated PD law:

  ```
  tau = -Kp * (q - q_des) - Kd * dq + g(q)
  ```

- **Cartesian mode** uses operational-space control: it computes a task-space wrench from the position/orientation error to the target, maps it to joint torques through the Jacobian and task-space inertia, and adds a null-space joint task plus gravity compensation.

You can tune the behavior by editing the gains near the top of `scripts/joint_impedance_control.py`:

- `Q_DES` — the desired/home joint pose.
- `KP_JOINT`, `KD_JOINT` — per-joint stiffness and damping (joint mode). Lower `KP_JOINT` = softer/more compliant.
- `impedance_pos`, `impedance_ori`, `damping_ratio` — Cartesian stiffness and damping.
- `Kp_null`, `Kd_null` — null-space joint gains (Cartesian mode).

## Controlling the robot in the MuJoCo viewer

The viewer uses the **left** mouse button to orbit the camera and the **right** mouse button to pan. To interact with the simulation, **hold `Ctrl`** and use the following:

### Pushing the arm (test impedance — both modes)

1. **Double-click** any robot link to select it (a selection box appears).
2. **`Ctrl` + right-drag** to push/translate the link, or **`Ctrl` + left-drag** to twist it.
3. Release. In joint mode the arm springs back to `Q_DES`; in Cartesian mode it yields and the end-effector keeps tracking the target. Lower the stiffness gains to feel a softer, more compliant response.

### Moving the target sphere (Cartesian mode)

The green sphere is the Cartesian target. The end-effector starts overlapping it, so to see tracking you need to move the sphere:

1. **Double-click** the green sphere to select it.
2. **`Ctrl` + right-drag** to **translate** it — the end-effector follows.
3. **`Ctrl` + left-drag** to **rotate** it — the end-effector reorients to match.

> **Note:** The green sphere only does something in `--mode cartesian`. In joint mode the arm tracks the fixed `Q_DES` pose and ignores the sphere.

### Mouse button cheat sheet

| Action | Mouse |
|--------|-------|
| Orbit camera | Left-drag |
| Pan camera | Right-drag |
| Zoom | Scroll wheel |
| Select a body | Double-click |
| **Translate** selected body | **`Ctrl` + right-drag** |
| **Rotate** selected body | **`Ctrl` + left-drag** |

> **Laptop trackpad tip:** "Right-drag" is the key to *translating* the target or pushing the arm sideways. If your trackpad has no right button enabled, translation will appear impossible and you'll only be able to rotate. Enable two-finger/right-click in your trackpad settings, or use an external mouse.
