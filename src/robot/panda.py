# src/robot/panda.py
# =============================================================================
# Franka Panda Robot Controller
#
# Loads the Panda URDF from pybullet_data and provides:
#   - Joint-space motion via position control
#   - Cartesian end-effector positioning via IK
#   - Configurable home, pre-grasp, and grasp poses
#   - Force/torque reading at the end-effector
#   - Real-time joint state logging
#
# The Panda has 7 arm joints (joint indices 0-6) and 2 finger joints
# (indices 9 and 10 in the standard pybullet URDF).
# =============================================================================

import numpy as np
import pybullet as p
import pybullet_data
import time


# Joint indices in the Panda URDF
PANDA_ARM_JOINTS     = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS  = [9, 10]
PANDA_EE_LINK        = 11   # end-effector link index

# Joint limits (from Franka documentation)
PANDA_JOINT_LIMITS_LOW  = [-2.897, -1.763, -2.897, -3.072,
                            -2.897, -0.018, -2.897]
PANDA_JOINT_LIMITS_HIGH = [ 2.897,  1.763,  2.897, -0.070,
                             2.897,  3.752,  2.897]

# Named joint configurations (radians)
# These were computed offline using IK for the table workspace
JOINT_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Pre-grasp pose: hovering above the work table center
JOINT_PREGRASP = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]

# Gripper width limits (meters)
GRIPPER_OPEN_WIDTH   = 0.08   # 80mm fully open
GRIPPER_CLOSED_WIDTH = 0.01   # 10mm near-closed

# Controller parameters
POSITION_GAIN  = 0.03
VELOCITY_GAIN  = 1.0
MAX_FORCE      = 240.0   # N per joint


class PandaRobot:
    """
    Franka Panda robot interface for PyBullet.

    Provides position-controlled joint motion, Cartesian IK,
    gripper control, and end-effector force sensing.

    Parameters
    ----------
    physics_client : int    PyBullet client ID
    base_position  : list   [x, y, z] robot base in world frame
    base_orientation : list Quaternion [x,y,z,w] (default: identity)
    """

    def __init__(self, physics_client,
                 base_position=[0.0, 0.0, 0.63],
                 base_orientation=[0, 0, 0, 1]):

        self.client      = physics_client
        self.base_pos    = base_position
        self.base_orn    = base_orientation
        self.robot_id    = None
        self._load()

    def _load(self):
        """Load Panda URDF from pybullet_data."""
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client
        )

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
            physicsClientId=self.client
        )

        # Confirm URDF loaded successfully before doing anything else
        if self.robot_id is None:
            raise RuntimeError(
                "Failed to load franka_panda/panda.urdf. "
                "Check that pybullet_data is installed correctly."
            )

        # Enable force-torque sensor on wrist joint AFTER confirming robot_id
        p.enableJointForceTorqueSensor(
            self.robot_id, 6, True,
            physicsClientId=self.client
        )

        # Set friction on finger pads
        for joint in PANDA_FINGER_JOINTS:
            p.changeDynamics(
                self.robot_id, joint,
                lateralFriction=2.0,
                physicsClientId=self.client
            )

        # Move to home position
        self.reset_to_home()

        print(f"Panda loaded (body ID={self.robot_id}). "
            f"Base at {self.base_pos}")

    # -------------------------------------------------------------------------
    # Joint control
    # -------------------------------------------------------------------------

    def reset_to_home(self):
        """
        Instantly teleport all joints to home configuration.
        Use only at scene initialization — not during simulation.
        """
        for i, joint_idx in enumerate(PANDA_ARM_JOINTS):
            p.resetJointState(
                self.robot_id, joint_idx,
                JOINT_HOME[i],
                physicsClientId=self.client
            )
        self.open_gripper(instant=True)

    def set_joint_positions(self, target_positions, max_force=MAX_FORCE):
        """
        Command all arm joints to target positions using position control.
        Call this every timestep while moving.

        Parameters
        ----------
        target_positions : list of 7 floats   Target joint angles (radians)
        max_force        : float              Maximum torque per joint (N·m)
        """
        p.setJointMotorControlArray(
            self.robot_id,
            PANDA_ARM_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            positionGains=[POSITION_GAIN] * 7,
            velocityGains=[VELOCITY_GAIN] * 7,
            forces=[max_force] * 7,
            physicsClientId=self.client
        )

    def move_to_joint_config(self, target_joints, threshold=0.02,
                              max_steps=500, dt=0.002, gui=False):
        """
        Block until joints reach target configuration or max_steps exceeded.

        Parameters
        ----------
        target_joints : list of 7 floats   Target joint angles (radians)
        threshold     : float              Joint error convergence threshold
        max_steps     : int                Maximum simulation steps to wait
        dt            : float              Simulation timestep
        gui           : bool               If True, add real-time sleep

        Returns
        -------
        bool   True if converged, False if timed out
        """
        for step in range(max_steps):
            self.set_joint_positions(target_joints)
            p.stepSimulation(physicsClientId=self.client)
            if gui:
                time.sleep(dt)

            # Check convergence
            current = self.get_joint_positions()
            error   = np.max(np.abs(
                np.array(current) - np.array(target_joints)
            ))
            if error < threshold:
                return True

        return False   # timed out

    def move_to_cartesian(self, target_pos, target_orn=None,
                           threshold=0.005, max_steps=500,
                           dt=0.002, gui=False):
        """
        Move end-effector to a Cartesian position using IK.

        Parameters
        ----------
        target_pos  : list [x, y, z]           World-frame target position
        target_orn  : list [x,y,z,w] or None   Target orientation quaternion
        threshold   : float                     Position error threshold (m)
        max_steps   : int
        dt          : float
        gui         : bool

        Returns
        -------
        bool   True if reached target
        """
        if target_orn is None:
            # Default: gripper pointing downward
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        for step in range(max_steps):
            # Compute IK
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                PANDA_EE_LINK,
                target_pos,
                target_orn,
                lowerLimits=PANDA_JOINT_LIMITS_LOW,
                upperLimits=PANDA_JOINT_LIMITS_HIGH,
                jointRanges=[5.8, 3.5, 5.8, 3.0, 5.8, 3.77, 5.8],
                restPoses=JOINT_HOME,
                physicsClientId=self.client
            )

            self.set_joint_positions(list(joint_poses[:7]))
            p.stepSimulation(physicsClientId=self.client)
            if gui:
                time.sleep(dt)

            # Check end-effector position
            ee_pos, _ = self.get_ee_pose()
            error = np.linalg.norm(
                np.array(ee_pos) - np.array(target_pos)
            )
            if error < threshold:
                return True

        return False

    # -------------------------------------------------------------------------
    # Gripper control
    # -------------------------------------------------------------------------

    def open_gripper(self, instant=False, steps=60, dt=0.002, gui=False):
        """Open the Panda fingers to full width."""
        self._set_gripper(GRIPPER_OPEN_WIDTH,
                          instant=instant, steps=steps, dt=dt, gui=gui)

    def close_gripper(self, width=GRIPPER_CLOSED_WIDTH,
                      instant=False, steps=60, dt=0.002, gui=False):
        """Close the Panda fingers to the specified width."""
        self._set_gripper(width,
                          instant=instant, steps=steps, dt=dt, gui=gui)

    def _set_gripper(self, width, instant=False, steps=60,
                     dt=0.002, gui=False):
        """Set both finger joints to half the desired total width."""
        half = width / 2.0
        if instant:
            for joint in PANDA_FINGER_JOINTS:
                p.resetJointState(
                    self.robot_id, joint, half,
                    physicsClientId=self.client
                )
            return

        for _ in range(steps):
            p.setJointMotorControlArray(
                self.robot_id,
                PANDA_FINGER_JOINTS,
                p.POSITION_CONTROL,
                targetPositions=[half, half],
                forces=[40.0, 40.0],
                physicsClientId=self.client
            )
            p.stepSimulation(physicsClientId=self.client)
            if gui:
                time.sleep(dt)

    def attach_object(self, object_id, ee_to_obj_pos=[0, 0, 0.05]):
        """
        Create a fixed constraint between end-effector and object.
        Used after gripper close to prevent slipping.

        Parameters
        ----------
        object_id    : int    PyBullet body ID of object to grasp
        ee_to_obj_pos: list   Offset from EE link to object center

        Returns
        -------
        int   Constraint ID (store this to remove later)
        """
        constraint_id = p.createConstraint(
            self.robot_id,          # parentBodyUniqueId
            PANDA_EE_LINK,          # parentLinkIndex
            object_id,              # childBodyUniqueId
            -1,                     # childLinkIndex (-1 = base)
            p.JOINT_FIXED,          # jointType
            [0, 0, 0],              # jointAxis
            ee_to_obj_pos,          # parentFramePosition
            [0, 0, 0],              # childFramePosition
            physicsClientId=self.client
        )
        return constraint_id

    def detach_object(self, constraint_id):
        """Remove a previously created grasp constraint."""
        p.removeConstraint(constraint_id,
                           physicsClientId=self.client)

    # -------------------------------------------------------------------------
    # State reading
    # -------------------------------------------------------------------------

    def get_joint_positions(self):
        """Return current arm joint positions as a list of 7 floats."""
        states = p.getJointStates(
            self.robot_id, PANDA_ARM_JOINTS,
            physicsClientId=self.client
        )
        return [s[0] for s in states]

    def get_joint_velocities(self):
        """Return current arm joint velocities."""
        states = p.getJointStates(
            self.robot_id, PANDA_ARM_JOINTS,
            physicsClientId=self.client
        )
        return [s[1] for s in states]

    def get_ee_pose(self):
        """
        Return end-effector world-frame position and orientation.

        Returns
        -------
        pos : list [x, y, z]
        orn : list [x, y, z, w]
        """
        state = p.getLinkState(
            self.robot_id,
            PANDA_EE_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.client
        )
        return list(state[4]), list(state[5])

    def get_ee_force(self):
        """
        Estimate end-effector force from wrist joint reaction forces.

        PyBullet does not provide a direct 6-axis force-torque sensor
        without a custom URDF. We approximate by reading joint 6
        reaction forces, which are the closest available proxy.

        Returns
        -------
        float   Resultant force magnitude (N) — approximate
        """
        joint_state = p.getJointState(
            self.robot_id, 6,
            physicsClientId=self.client
        )
        # joint_state[2] is the joint reaction forces: [Fx,Fy,Fz,Mx,My,Mz]
        reaction = joint_state[2]
        force_mag = float(np.linalg.norm(reaction[:3]))
        return force_mag

    def get_gripper_width(self):
        """Return current total gripper opening width (meters)."""
        states = p.getJointStates(
            self.robot_id, PANDA_FINGER_JOINTS,
            physicsClientId=self.client
        )
        return states[0][0] + states[1][0]

    # -------------------------------------------------------------------------
    # Debug visualization
    # -------------------------------------------------------------------------

    def draw_ee_frame(self, length=0.05):
        """Draw XYZ axes at the end-effector for debugging."""
        ee_pos, ee_orn = self.get_ee_pose()
        rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        colors = [[1,0,0], [0,1,0], [0,0,1]]
        for i, color in enumerate(colors):
            end = np.array(ee_pos) + length * rot[:, i]
            p.addUserDebugLine(
                ee_pos, end.tolist(), color,
                lineWidth=2,
                physicsClientId=self.client
            )