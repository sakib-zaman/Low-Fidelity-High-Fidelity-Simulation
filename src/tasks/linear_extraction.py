import time
import numpy as np
import pybullet as p

from robot.panda import PandaRobot, PANDA_EE_LINK, JOINT_HOME
from simulation.scene import Scene, WORK_SURFACE_Z
from simulation.connector import Connector
from objects.pcb import PCBObject
from objects.gpu import GPUObject


class LinearExtractionTask:
    def __init__(
        self,
        physics_client,
        task_name,
        fidelity,
        strength,
        friction,
        misalignment,
        gui=False,
        dt=1.0 / 240.0,
    ):
        self.client = physics_client
        self.task_name = task_name
        self.fidelity = fidelity
        self.strength = strength
        self.friction = friction
        self.misalignment = misalignment
        self.gui = gui
        self.dt = dt

        self.scene = None
        self.robot = None
        self.product = None
        self.connectors = []
        self.grasp_constraint = None

    def setup(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=120,
            physicsClientId=self.client,
        )

        self.scene = Scene(self.client, gui=self.gui)
        self.robot = PandaRobot(
            self.client,
            base_position=[0.0, 0.0, WORK_SURFACE_Z + 0.05],
        )

        if self.task_name == "pcb":
            self.product = PCBObject(
                self.client,
                position=[0.52, 0.05, WORK_SURFACE_Z + 0.10],
                friction=self.friction,
                misalignment=self.misalignment,
                fidelity=self.fidelity,
            )
        elif self.task_name == "gpu":
            self.product = GPUObject(
                self.client,
                position=[0.50, 0.05, WORK_SURFACE_Z + 0.12],
                friction=self.friction,
                misalignment=self.misalignment,
                fidelity=self.fidelity,
            )
        else:
            raise ValueError(self.task_name)

        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client)

        self._build_connectors()

    def _build_connectors(self):
        moving_body = self.product.get_object_id()
        fixture_body = self.product.chassis_id

        if self.fidelity == "LF":
            anchors = [[0.0, 0.0, 0.0]]
        elif self.task_name == "pcb":
            anchors = [
                [-0.050, 0.0, -0.035],
                [-0.025, 0.0, -0.035],
                [0.000, 0.0, -0.035],
                [0.025, 0.0, -0.035],
                [0.050, 0.0, -0.035],
                [-0.050, 0.0, 0.035],
                [0.050, 0.0, 0.035],
                [-0.030, 0.0, 0.000],
                [0.030, 0.0, 0.000],
                [0.000, 0.0, 0.035],
            ]
        else:
            anchors = [
                [-0.080, 0.0, -0.045],
                [-0.040, 0.0, -0.045],
                [0.000, 0.0, -0.045],
                [0.040, 0.0, -0.045],
                [0.080, 0.0, -0.045],
                [-0.080, 0.0, 0.045],
                [0.080, 0.0, 0.045],
                [-0.060, 0.0, 0.000],
                [0.060, 0.0, 0.000],
                [0.080, 0.0, -0.030],
            ]

        count = len(anchors)

        for index, anchor in enumerate(anchors):
            self.connectors.append(
                Connector(
                    self.client,
                    moving_body,
                    fixture_body,
                    anchor,
                    anchor,
                    k=4000.0 / count,
                    c=35.0 / count,
                    strength=self.strength / count,
                    connector_id=index,
                )
            )

    def _move(self, target_position, target_orientation, steps=400):
        for _ in range(steps):
            joints = p.calculateInverseKinematics(
                self.robot.robot_id,
                PANDA_EE_LINK,
                target_position,
                target_orientation,
                physicsClientId=self.client,
            )
            self.robot.set_joint_positions(joints[:7])
            p.stepSimulation(physicsClientId=self.client)

            if self.gui:
                time.sleep(self.dt)

            current_position, _ = self.robot.get_ee_pose()

            if np.linalg.norm(
                np.asarray(current_position)
                - np.asarray(target_position)
            ) < 0.01:
                return True

        return False

    def _grasp(self):
        object_position, object_orientation = (
            self.product.get_pose()
        )
        ee_position, ee_orientation = self.robot.get_ee_pose()

        inverse_ee_position, inverse_ee_orientation = (
            p.invertTransform(
                ee_position,
                ee_orientation,
            )
        )

        relative_position, relative_orientation = (
            p.multiplyTransforms(
                inverse_ee_position,
                inverse_ee_orientation,
                object_position,
                object_orientation,
            )
        )

        self.grasp_constraint = p.createConstraint(
            self.robot.robot_id,
            PANDA_EE_LINK,
            self.product.get_object_id(),
            -1,
            p.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            relative_position,
            [0.0, 0.0, 0.0],
            relative_orientation,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

        for connector in self.connectors:
            connector.rebase()

    def run(self):
        start_time = time.perf_counter()
        self.setup()

        object_position, _ = self.product.get_pose()

        # Gripper points toward the product face.
        orientation = p.getQuaternionFromEuler(
            [np.pi / 2.0, 0.0, 0.0]
        )

        pregrasp = [
            object_position[0],
            object_position[1] - 0.12,
            object_position[2],
        ]
        grasp = [
            object_position[0],
            object_position[1] - 0.02,
            object_position[2],
        ]

        self.robot.open_gripper(instant=True)
        self._move(pregrasp, orientation)
        self._move(grasp, orientation)
        self.robot.close_gripper(
            width=0.04,
            steps=40,
            dt=self.dt,
            gui=self.gui,
        )
        self._grasp()

        initial_position = np.asarray(
            self.product.get_pose()[0]
        )

        peak_force = 0.0
        first_failure_time = None
        final_failure_time = None
        success = False

        extraction_steps = int(
            0.13 / (0.025 * self.dt)
        )

        for step in range(extraction_steps):
            simulation_time = step * self.dt
            target = [
                grasp[0],
                grasp[1] - 0.025 * simulation_time,
                grasp[2],
            ]

            joints = p.calculateInverseKinematics(
                self.robot.robot_id,
                PANDA_EE_LINK,
                target,
                orientation,
                physicsClientId=self.client,
            )
            self.robot.set_joint_positions(joints[:7])

            forces = [
                connector.step(self.dt, simulation_time)
                for connector in self.connectors
            ]
            peak_force = max(
                peak_force,
                float(np.sum(forces)),
            )

            p.stepSimulation(physicsClientId=self.client)

            if self.gui:
                time.sleep(self.dt)

            failed_count = sum(
                connector.failed
                for connector in self.connectors
            )

            if failed_count and first_failure_time is None:
                first_failure_time = simulation_time

            if (
                failed_count == len(self.connectors)
                and final_failure_time is None
            ):
                final_failure_time = simulation_time

            current_position = np.asarray(
                self.product.get_pose()[0]
            )
            displacement = np.linalg.norm(
                current_position - initial_position
            )

            if (
                failed_count == len(self.connectors)
                and displacement >= 0.08
            ):
                success = True
                break

        placement_success = False

        if success:
            place_position = [0.72, -0.20, WORK_SURFACE_Z + 0.08]
            self._move(place_position, orientation)

            p.removeConstraint(
                self.grasp_constraint,
                physicsClientId=self.client,
            )
            self.grasp_constraint = None
            self.robot.open_gripper(
                steps=40,
                dt=self.dt,
                gui=self.gui,
            )

            placement_success = True

        self.robot.move_to_joint_config(
            JOINT_HOME,
            gui=self.gui,
            dt=self.dt,
        )

        return {
            "task": self.task_name,
            "fidelity": self.fidelity,
            "success": success,
            "placement_success": placement_success,
            "peak_force": peak_force,
            "safety_violated": False,
            "first_failure_time": first_failure_time,
            "final_failure_time": final_failure_time,
            "wall_time": time.perf_counter() - start_time,
        }