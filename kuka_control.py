import os
import time
import argparse
import numpy as np
import mujoco

# Import constants and control functions from the mujoco script
from scripts.joint_impedance_control import (
    SCENE_PATH, Q_DES, KP_JOINT, KD_JOINT,
    SITE_NAME, MOCAP_NAME,
    joint_impedance_torque, cartesian_opspace_torque
)

class KukaHardwareInterface:
    """
    Drake LCM interface for KUKA FRI.
    Requires running Drake's `drake-fri-client` (or similar bridge) 
    to bridge the physical robot UDP packets to LCM messages on localhost.
    """
    def __init__(self):
        print("Initializing Drake LCM connection to KUKA arm...")
        try:
            from pydrake.lcm import DrakeLcm
            from pydrake.lcmt_iiwa_status import lcmt_iiwa_status
            from pydrake.lcmt_iiwa_command import lcmt_iiwa_command
        except ImportError:
            print("\nERROR: pydrake is not installed or lcmt_iiwa_status not found.")
            print("Make sure you install drake: pip install drake\n")
            raise

        self.lcmt_iiwa_status = lcmt_iiwa_status
        self.lcm = DrakeLcm()
        self.status_msg = None
        
        self.command_msg = lcmt_iiwa_command()
        self.command_msg.num_joints = 7
        self.command_msg.joint_position = [0.0] * 7
        
        def status_callback(raw_data):
            # Decode the LCM byte array into the status struct
            self.status_msg = self.lcmt_iiwa_status.decode(raw_data)
            
        # Subscribe to the status channel published by drake-fri-client
        self.lcm.Subscribe("IIWA_STATUS", status_callback)
        
        # Start a background thread to pump LCM messages
        self.lcm.StartReceiveThread()
        
        print("Waiting for first IIWA_STATUS message from FRI bridge...")
        while self.status_msg is None:
            time.sleep(0.1)
        print("Received first status message. Ready to control.")

    def get_joint_state(self):
        """Read the current joint positions (q) and velocities (dq) from the latest LCM message."""
        q = np.array(self.status_msg.joint_position_measured)
        dq = np.array(self.status_msg.joint_velocity_estimated)
        return q, dq
        
    def send_joint_torques(self, torques):
        """Command the calculated joint torques to the physical arm via LCM."""
        self.command_msg.utime = int(time.time() * 1e6)
        
        # In torque mode, we often need to also pass the desired position.
        # Here we just pass the current measured position as a fallback, 
        # but the KUKA cabinet will listen to joint_torque if in impedance mode.
        self.command_msg.joint_position = list(self.status_msg.joint_position_measured)
        self.command_msg.joint_torque = torques.tolist()
        
        # Publish the command back to the FRI bridge
        self.lcm.Publish("IIWA_COMMAND", self.command_msg.encode())


def main():
    parser = argparse.ArgumentParser(description="Physical KUKA impedance control")
    parser.add_argument("--mode", type=str, choices=["joint", "cartesian"], default="joint",
                        help="Control mode: 'joint' or 'cartesian'")
    args = parser.parse_args()

    # Load MuJoCo model to act as our dynamics and kinematics engine
    # We will use it to compute gravity compensation, Jacobians, and inertia matrices
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    robot = KukaHardwareInterface()
    
    print(f"Loaded dynamic model from {SCENE_PATH}")
    print(f"Control Mode: {args.mode.upper()}")
    
    # If using Cartesian mode, define a target pose (mocap)
    # In a real scenario, this target might come from a trajectory generator or ROS topic
    if args.mode == "cartesian":
        mocap_id = model.body(MOCAP_NAME).mocapid[0]
        # Set a fixed target for demonstration (e.g., 0.5m high, 0.4m forward)
        data.mocap_pos[mocap_id] = np.array([0.4, 0.0, 0.5])
        data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0]) # Pointing downwards
    
    control_rate = 1000 # Hz (Common for impedance control on KUKA FRI)
    dt = 1.0 / control_rate
    
    print("Starting control loop... Press Ctrl+C to stop.")
    try:
        while True:
            step_start = time.time()
            
            # 1. Read state from the physical robot
            q, dq = robot.get_joint_state()
            
            # 2. Update the MuJoCo state with the real robot's state
            data.qpos[:7] = q
            data.qvel[:7] = dq
            
            # Run forward dynamics to update kinematic/dynamic quantities (e.g. gravity, Jacobians)
            mujoco.mj_forward(model, data)
            
            # 3. Compute impedance control torques using the functions from our MuJoCo script
            if args.mode == "joint":
                # joint_impedance_torque calculates: -Kp*(q - q_des) - Kd*dq + gravity
                tau = joint_impedance_torque(model, data, Q_DES, KP_JOINT, KD_JOINT)
            else:
                # cartesian_opspace_torque calculates operational space torques
                tau = cartesian_opspace_torque(model, data, SITE_NAME, MOCAP_NAME, Q_DES)
                
            # 4. Command the torques to the physical robot
            # We clip the torques to the actuator limits defined in the MuJoCo model for safety
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            tau_safe = np.clip(tau, lo, hi)
            
            robot.send_joint_torques(tau_safe)
            
            # 5. Enforce control rate
            elapsed = time.time() - step_start
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
                
    except KeyboardInterrupt:
        print("\nStopping control loop.")

if __name__ == "__main__":
    main()
