import numpy as np
import pybullet as p


class Connector:
    """
    Breakable tensile connector.

    Force is based on separation normal to the HDD interface. Lateral
    misalignment is included in the reference configuration and therefore
    does not create an artificial initial extension.
    """

    def __init__(
        self,
        physics_client,
        body_lid,
        body_housing,
        anchor_lid_local,
        anchor_housing_local,
        k,
        c,
        strength,
        connector_id=0,
    ):
        self.client = physics_client
        self.body_lid = body_lid
        self.body_housing = body_housing

        self.anchor_lid_local = np.asarray(
            anchor_lid_local, dtype=float
        )
        self.anchor_housing_local = np.asarray(
            anchor_housing_local, dtype=float
        )

        self.k = float(k)
        self.c = float(c)
        self.strength = float(strength)
        self.connector_id = int(connector_id)

        self.failed = False
        self.failure_time = None
        self.peak_force = 0.0

        self.rest_vector = np.zeros(3)
        self.previous_extension = 0.0
        self.initialized = False

        self.rebase()

    def _rotation_matrix(self, quaternion):
        return np.asarray(
            p.getMatrixFromQuaternion(quaternion)
        ).reshape(3, 3)

    def _get_world_anchors(self):
        lid_pos, lid_orn = p.getBasePositionAndOrientation(
            self.body_lid,
            physicsClientId=self.client,
        )
        housing_pos, housing_orn = p.getBasePositionAndOrientation(
            self.body_housing,
            physicsClientId=self.client,
        )

        lid_rotation = self._rotation_matrix(lid_orn)
        housing_rotation = self._rotation_matrix(housing_orn)

        lid_anchor_world = (
            np.asarray(lid_pos)
            + lid_rotation @ self.anchor_lid_local
        )
        housing_anchor_world = (
            np.asarray(housing_pos)
            + housing_rotation @ self.anchor_housing_local
        )

        return (
            lid_anchor_world,
            housing_anchor_world,
            housing_rotation,
        )

    def rebase(self):
        """
        Define the current assembled pose as zero extension.

        Call this after the grasp constraint is created and stabilized.
        """
        lid_anchor, housing_anchor, _ = self._get_world_anchors()

        self.rest_vector = lid_anchor - housing_anchor
        self.previous_extension = 0.0
        self.initialized = True

    def step(self, dt, current_time):
        if self.failed:
            return 0.0

        if not self.initialized:
            self.rebase()
            return 0.0

        lid_anchor, housing_anchor, housing_rotation = (
            self._get_world_anchors()
        )

        current_vector = lid_anchor - housing_anchor
        displacement = current_vector - self.rest_vector

        # Local +Z axis of the housing is the interface separation normal.
        separation_normal = housing_rotation[:, 2]

        # Only opening displacement generates attachment force.
        extension = max(
            float(np.dot(displacement, separation_normal)),
            0.0,
        )

        extension_rate = (
            extension - self.previous_extension
        ) / max(float(dt), 1e-6)

        # Prevent one-step numerical velocity spikes.
        extension_rate = float(
            np.clip(extension_rate, -0.25, 0.25)
        )

        force_magnitude = max(
            self.k * extension + self.c * extension_rate,
            0.0,
        )

        self.previous_extension = extension
        self.peak_force = max(
            self.peak_force,
            force_magnitude,
        )

        if force_magnitude >= self.strength:
            self.failed = True
            self.failure_time = float(current_time)
            return force_magnitude

        force_vector = force_magnitude * separation_normal

        p.applyExternalForce(
            self.body_lid,
            -1,
            (-force_vector).tolist(),
            lid_anchor.tolist(),
            p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        p.applyExternalForce(
            self.body_housing,
            -1,
            force_vector.tolist(),
            housing_anchor.tolist(),
            p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        return force_magnitude

    @property
    def is_intact(self):
        return not self.failed