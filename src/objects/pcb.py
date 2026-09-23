import pybullet as p


PCB_HALF = [0.060, 0.001, 0.040]
PCB_MASS = 0.090


class PCBObject:
    def __init__(
        self,
        client,
        position,
        friction,
        misalignment,
        fidelity,
    ):
        self.client = client
        self.position = position
        self.friction = friction
        self.misalignment = misalignment
        self.fidelity = fidelity

        self.chassis_id = None
        self.board_id = None
        self.detail_ids = []

        self._build()

    def _box(self, half, position, color, mass=0.0):
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half,
            physicsClientId=self.client,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=color,
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client,
        )

    def _build(self):
        x, y, z = self.position

        # Server chassis backplane.
        self.chassis_id = self._box(
            [0.075, 0.025, 0.055],
            [x, y + 0.025, z],
            [0.25, 0.27, 0.30, 1.0],
            mass=0.0,
        )

        board_orientation = p.getQuaternionFromEuler(
            [0.0, self.misalignment, 0.0]
        )

        board_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=PCB_HALF,
            physicsClientId=self.client,
        )
        board_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=PCB_HALF,
            rgbaColor=[0.04, 0.42, 0.12, 1.0],
            physicsClientId=self.client,
        )
        self.board_id = p.createMultiBody(
            baseMass=PCB_MASS,
            baseCollisionShapeIndex=board_collision,
            baseVisualShapeIndex=board_visual,
            basePosition=[x, y, z],
            baseOrientation=board_orientation,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.board_id,
            -1,
            lateralFriction=self.friction,
            physicsClientId=self.client,
        )

        # Components make the PCB visually recognizable.
        components = [
            ([-0.030, -0.003, 0.010], [0.012, 0.004, 0.012],
             [0.10, 0.10, 0.10, 1.0]),
            ([0.005, -0.003, 0.015], [0.015, 0.004, 0.010],
             [0.12, 0.12, 0.12, 1.0]),
            ([0.035, -0.003, -0.010], [0.010, 0.004, 0.016],
             [0.20, 0.20, 0.22, 1.0]),
        ]

        for offset, half, color in components:
            component = self._box(
                half,
                [
                    x + offset[0],
                    y + offset[1],
                    z + offset[2],
                ],
                color,
                mass=0.0,
            )
            self.detail_ids.append(component)

    def get_object_id(self):
        return self.board_id

    def get_pose(self):
        return p.getBasePositionAndOrientation(
            self.board_id,
            physicsClientId=self.client,
        )