# src/tasks/hdd_removal.py
import time
import numpy as np
import pybullet as p

from robot.panda import PandaRobot, PANDA_EE_LINK, JOINT_HOME
from objects.hdd import HDDObject, HDD_ANCHORS, HDD_STRENGTH_WEIGHTS, BASE_HALF, COVER_HALF
from simulation.connector import Connector
from simulation.scene import Scene, WORK_SURFACE_Z

PULL_VELOCITY     = 0.035   # m/s
PULL_DISTANCE     = 0.070   # 70mm lift
SUCCESS_CLEARANCE = 0.025   # 25mm lift required
SETTLE_STEPS      = 60

CONNECTOR_K       = 4000.0  # N/m
CONNECTOR_C       = 35.0    # N*s/m

MAX_EE_FORCE      = 150.0   # N
MAX_ROTATION_DEG  = 35.0    # deg

HDD_TABLE_POS     = [0.52, 0.0, WORK_SURFACE_Z]
ROBOT_BASE_Z      = WORK_SURFACE_Z + 0.05

class HDDRemovalTask:
    def __init__(self, physics_client, fidelity, S, mu, delta, gui=False, dt=1.0/120.0):
        self.client   = physics_client
        self.fidelity = fidelity
        self.S        = S
        self.mu       = mu
        self.delta    = delta
        self.gui      = gui
        self.dt       = dt

        self.scene            = None
        self.robot            = None
        self.hdd              = None
        self.connectors       = []
        self.grasp_constraint = None

        self.ee_to_cover_position = None
        self.ee_to_cover_orientation = None

    def setup(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=120, physicsClientId=self.client)

        self.scene = Scene(self.client, gui=self.gui)
        self.robot = PandaRobot(physics_client=self.client, base_position=[0.0, 0.0, ROBOT_BASE_Z])
        self.hdd   = HDDObject(physics_client=self.client, position=HDD_TABLE_POS, mu=self.mu, delta=self.delta, fidelity=self.fidelity)

        # Let parts settle under gravity
        for _ in range(SETTLE_STEPS):
            self.robot.set_joint_positions(self.robot.get_joint_positions())
            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(self.dt)

        # Build connectors at settled equilibrium
        self._build_connectors()

    def _build_connectors(self):
        self.connectors = []
        housing_top_z = BASE_HALF[2] # top face in housing frame

        if self.fidelity == 'LF':
            conn = Connector(
                self.client, self.hdd.cover_id, self.hdd.housing_id,
                anchor_lid_local=[0.0, 0.0, -COVER_HALF[2]],
                anchor_housing_local=[0.0, 0.0, housing_top_z],
                k=CONNECTOR_K, c=CONNECTOR_C, strength=self.S, connector_id=0
            )
            self.connectors.append(conn)
        else:
            for i, anchor in enumerate(HDD_ANCHORS):
                s_i = (self.S * HDD_STRENGTH_WEIGHTS[i]) / 8.0
                number_of_connectors = len(HDD_ANCHORS)

                connector = Connector(
                    self.client,
                    self.hdd.cover_id,
                    self.hdd.housing_id,
                    anchor_lid_local=anchor,
                    anchor_housing_local=[
                        anchor[0],
                        anchor[1],
                        housing_top_z,
                    ],
                    k=CONNECTOR_K / number_of_connectors,
                    c=CONNECTOR_C / number_of_connectors,
                    strength=(
                        self.S
                        * HDD_STRENGTH_WEIGHTS[i]
                        / number_of_connectors
                    ),
                    connector_id=i,
                )

    def _move_cartesian(self, target_pos, target_orn=None, threshold=0.008, max_steps=400):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        for _ in range(max_steps):
            joints = p.calculateInverseKinematics(
                self.robot.robot_id, PANDA_EE_LINK, target_pos, target_orn,
                physicsClientId=self.client
            )
            self.robot.set_joint_positions(list(joints[:7]))
            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(self.dt)
                if self.connectors:
                    self.hdd.draw_connectors(self.connectors)

            ee_pos, _ = self.robot.get_ee_pose()
            if np.linalg.norm(np.array(ee_pos) - np.array(target_pos)) < threshold:
                return True
        return False

    def _phase0_pregrasp(self):
        print("  [Phase 0] Moving to pre-grasp...")
        self.robot.open_gripper(instant=True)
        pre_pos = self.hdd.get_pregrasp_position()
        self._move_cartesian(pre_pos, max_steps=300)

    def _phase1_descend(self):
        print("  [Phase 1] Descending to grasp position...")
        grasp_pos = self.hdd.get_grasp_position()
        self._move_cartesian(grasp_pos, threshold=0.005, max_steps=350)

    def _phase2_grasp(self):
        print("  [Phase 2] Grasping cover...")

        self.robot.close_gripper(
            width=0.068,
            steps=40,
            dt=self.dt,
            gui=self.gui,
        )

        ee_state = p.getLinkState(
            self.robot.robot_id,
            PANDA_EE_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        ee_position = ee_state[4]
        ee_orientation = ee_state[5]

        cover_position, cover_orientation = (
            p.getBasePositionAndOrientation(
                self.hdd.cover_id,
                physicsClientId=self.client,
            )
        )

        inverse_ee_position, inverse_ee_orientation = (
            p.invertTransform(
                ee_position,
                ee_orientation,
            )
        )

        (
            self.ee_to_cover_position,
            self.ee_to_cover_orientation,
        ) = p.multiplyTransforms(
            inverse_ee_position,
            inverse_ee_orientation,
            cover_position,
            cover_orientation,
        )

        self.grasp_constraint = p.createConstraint(
            self.robot.robot_id,
            PANDA_EE_LINK,
            self.hdd.cover_id,
            -1,
            p.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            self.ee_to_cover_position,
            [0.0, 0.0, 0.0],
            self.ee_to_cover_orientation,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

        p.changeConstraint(
            self.grasp_constraint,
            maxForce=200.0,
            erp=0.2,
            physicsClientId=self.client,
        )

        # Stabilize the fixed grasp while the attachments are inactive.
        current_joints = self.robot.get_joint_positions()

        for _ in range(20):
            self.robot.set_joint_positions(current_joints)
            p.stepSimulation(physicsClientId=self.client)

            if self.gui:
                time.sleep(self.dt)

        # Establish zero attachment extension after grasp stabilization.
        for connector in self.connectors:
            connector.rebase()

        print("  [Phase 2] Grasp stabilized; connectors activated.")

    def _phase3_pull(self):
        print("  [Phase 3] Pulling cover upward...")
        grasp_pos    = self.hdd.get_grasp_position()
        init_cover_z = self.hdd.get_cover_position()[2]
        pull_steps   = int(PULL_DISTANCE / (PULL_VELOCITY * self.dt))
        target_orn   = p.getQuaternionFromEuler([np.pi, 0, 0])

        force_series    = []
        lid_z_series    = []
        time_series     = []
        success         = False
        safety_violated = False
        max_rotation    = 0.0
        peak_force      = 0.0

        for step in range(pull_steps):
            sim_time = step * self.dt
            target_z = grasp_pos[2] + PULL_VELOCITY * sim_time
            target   = [grasp_pos[0], grasp_pos[1], target_z]

            joints = p.calculateInverseKinematics(
                self.robot.robot_id, PANDA_EE_LINK, target, target_orn,
                physicsClientId=self.client
            )
            self.robot.set_joint_positions(list(joints[:7]))

            # Step physical connectors
            connector_forces = [
                connector.step(self.dt, sim_time)
                for connector in self.connectors
            ]

            total_attachment_force = float(
                np.sum(connector_forces)
            )

            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(self.dt)
                self.hdd.draw_connectors(self.connectors)

            cover_pos = self.hdd.get_cover_position()
            cover_orn = self.hdd.get_cover_orientation()

            # Read force
            measured_force = total_attachment_force
            peak_force = max(peak_force, measured_force)

            force_series.append(measured_force)
            lid_z_series.append(cover_pos[2])
            time_series.append(sim_time)

            euler = p.getEulerFromQuaternion(cover_orn)
            rot_z = abs(np.degrees(euler[2]))
            max_rotation = max(max_rotation, rot_z)

            # Safety check
            if measured_force > MAX_EE_FORCE or rot_z > MAX_ROTATION_DEG:
                safety_violated = True
                print(f"  [Phase 3] Safety violation: F={measured_force:.1f}N, Rot={rot_z:.1f} deg")
                break

            # Success check
            all_failed = all(c.failed for c in self.connectors)
            lift = cover_pos[2] - init_cover_z
            if all_failed and lift >= SUCCESS_CLEARANCE:
                success = True
                print(f"  [Phase 3] Separation success at t={sim_time:.3f}s (Lift={lift*1000:.1f}mm)")
                break

        return (success, safety_violated, peak_force, max_rotation, force_series, lid_z_series, time_series)

    def _phase4_place(self):
        print("  [Phase 4] Transporting cover to placement area...")

        desired_cover_orientation = p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        )

        # Keep the placement inside the right side of the workbench.
        place_cover_position = [
            0.72,
            -0.18,
            WORK_SURFACE_Z + COVER_HALF[2] + 0.003,
        ]

        # First move above the placement location.
        transit_cover_position = [
            place_cover_position[0],
            place_cover_position[1],
            place_cover_position[2] + 0.12,
        ]

        transit_ee_position, transit_ee_orientation = (
            self._ee_pose_for_cover_pose(
                transit_cover_position,
                desired_cover_orientation,
            )
        )

        reached_transit = self._move_cartesian(
            transit_ee_position,
            target_orn=transit_ee_orientation,
            threshold=0.012,
            max_steps=500,
        )

        if not reached_transit:
            print("  [Phase 4] Warning: transit pose was not reached.")
            return False

        print("  [Phase 4] Lowering cover onto table...")

        place_ee_position, place_ee_orientation = (
            self._ee_pose_for_cover_pose(
                place_cover_position,
                desired_cover_orientation,
            )
        )

        reached_place = self._move_cartesian(
            place_ee_position,
            target_orn=place_ee_orientation,
            threshold=0.008,
            max_steps=500,
        )

        if not reached_place:
            print("  [Phase 4] Warning: placement pose was not reached.")
            return False

        return True

    def _phase5_release(self):
        print("  [Phase 5] Releasing cover...")

        # Remove the fixed grasp before opening the fingers.
        if self.grasp_constraint is not None:
            p.removeConstraint(
                self.grasp_constraint,
                physicsClientId=self.client,
            )
            self.grasp_constraint = None

        self.robot.open_gripper(
            instant=False,
            steps=40,
            dt=self.dt,
            gui=self.gui,
        )

        # Allow the released cover to settle on the workbench.
        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client)

            if self.gui:
                time.sleep(self.dt)

        cover_position = self.hdd.get_cover_position()
        expected_z = WORK_SURFACE_Z + COVER_HALF[2]

        placed_correctly = (
            abs(cover_position[2] - expected_z) < 0.015
            and cover_position[0] > 0.62
        )

        print(
            "  [Phase 5] Cover position after release: "
            f"{[round(value, 4) for value in cover_position]}"
        )
        print(
            f"  [Phase 5] Placement verified: {placed_correctly}"
        )

        # Retract vertically without touching the released cover.
        ee_position, ee_orientation = self.robot.get_ee_pose()
        retract_position = [
            ee_position[0],
            ee_position[1],
            ee_position[2] + 0.10,
        ]

        self._move_cartesian(
            retract_position,
            target_orn=ee_orientation,
            threshold=0.012,
            max_steps=300,
        )

        return placed_correctly

    def _phase6_return(self):
        print("  [Phase 6] Returning to home position...")
        self.robot.move_to_joint_config(JOINT_HOME, gui=self.gui, dt=self.dt, max_steps=400)

    def run(self):
        wall_start = time.perf_counter()
        self.setup()
        self._phase0_pregrasp()
        self._phase1_descend()
        self._phase2_grasp()

        (success, safety_violated, peak_force,
         max_rotation, force_series, lid_z_series, time_series) = self._phase3_pull()

        placement_success = False

        if success:
            reached_place = self._phase4_place()

            if reached_place:
                placement_success = self._phase5_release()

        self._phase6_return()
        wall_time = time.perf_counter() - wall_start

        failure_sequence = sorted(
            [(c.connector_id, c.failure_time) for c in self.connectors if c.failed],
            key=lambda x: (x[1] is None, x[1])
        )

        return {
            "task":                "hdd_removal",
            "fidelity":            self.fidelity,
            "success":             success,
            "peak_force":          peak_force,
            "safety_violated":     safety_violated,
            "max_rotation_deg":    max_rotation,
            "wall_time":           wall_time,
            "force_series":        np.array(force_series),
            "lid_z_series":        np.array(lid_z_series),
            "time_series":         np.array(time_series),
            "failure_sequence":    failure_sequence,
            "placement_success":   placement_success,
            "n_connectors_failed": sum(1 for c in self.connectors if c.failed),
        }

    def _ee_pose_for_cover_pose(
        self,
        desired_cover_position,
        desired_cover_orientation,
    ):
        """
        Convert a desired cover pose into the end-effector pose required
        to achieve it while maintaining the fixed grasp transform.

        T_world_cover = T_world_ee * T_ee_cover
        T_world_ee = T_world_cover * inverse(T_ee_cover)
        """
        inverse_relative_position, inverse_relative_orientation = (
            p.invertTransform(
                self.ee_to_cover_position,
                self.ee_to_cover_orientation,
            )
        )

        target_ee_position, target_ee_orientation = (
            p.multiplyTransforms(
                desired_cover_position,
                desired_cover_orientation,
                inverse_relative_position,
                inverse_relative_orientation,
            )
        )

        return list(target_ee_position), list(target_ee_orientation)