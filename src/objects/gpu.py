import pybullet as p


GPU_HALF = [0.090, 0.006, 0.050]
GPU_MASS = 0.750


class GPUObject:
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
        self.gpu_id = None
        self.latch_id = None
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

        self.chassis_id = self._box(
            [0.110, 0.030, 0.070],
            [x, y + 0.030, z],
            [0.20, 0.22, 0.25, 1.0],
        )

        orientation = p.getQuaternionFromEuler(
            [self.misalignment, 0.0, 0.0]
        )

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=GPU_HALF,
            physicsClientId=self.client,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=GPU_HALF,
            rgbaColor=[0.06, 0.12, 0.08, 1.0],
            physicsClientId=self.client,
        )

        self.gpu_id = p.createMultiBody(
            baseMass=GPU_MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            baseOrientation=orientation,
            physicsClientId=self.client,
        )

        # PCIe retention latch.
        self.latch_id = self._box(
            [0.012, 0.010, 0.005],
            [x + 0.080, y - 0.005, z - 0.045],
            [0.85, 0.75, 0.15, 1.0],
        )

        # Cooling shroud and two fan hubs.
        self._box(
            [0.075, 0.009, 0.040],
            [x, y - 0.010, z],
            [0.10, 0.10, 0.11, 1.0],
        )

        for fan_x in [x - 0.035, x + 0.035]:
            visual_fan = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.022,
                length=0.004,
                rgbaColor=[0.03, 0.03, 0.03, 1.0],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_fan,
                basePosition=[fan_x, y - 0.020, z],
                baseOrientation=p.getQuaternionFromEuler(
                    [p.getQuaternionFromEuler([0, 0, 0])[0], 0, 0]
                ),
                physicsClientId=self.client,
            )

    def get_object_id(self):
        return self.gpu_id

    def get_pose(self):
        return p.getBasePositionAndOrientation(
            self.gpu_id,
            physicsClientId=self.client,
        )