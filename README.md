# Kuka Impedance Control

This project simulates joint-space impedance control for a KUKA LBR iiwa14 robot arm using MuJoCo. The simulation demonstrates a gravity-compensated PD control law that makes the arm behave compliantly. You can interact with the arm in real-time by applying external forces in the viewer.

## Installation

To install the required dependencies for this project, use the provided `requirements.txt` file. It's recommended to install these into a Python virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

You can run the simulation using the provided Python script:

```bash
python3 scripts/joint_impedance_control.py
```

Once the simulation launches, you can interact with the robot arm:
- The arm will hold its initial configuration while compensating for gravity.
- **Ctrl + Drag** any link of the robot in the viewer to apply external forces and physically feel the impedance controller reacting to the disturbance.