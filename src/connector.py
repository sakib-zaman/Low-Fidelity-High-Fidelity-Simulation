# connector.py
# =============================================================================
# Breakable spring-damper connector for PyBullet.
#
# Physical model:
#   F_connector = k * extension + c * extension_rate
#
# The connector fails permanently when F_connector >= strength (N).
# Forces are applied at the world-frame anchor positions, so off-center
# connectors naturally produce torques on the lid — this is physically
# correct and is why the HF model produces rotation that LF cannot.
#
# PyBullet does not support breakable native constraints, so we compute
# and inject forces manually via applyExternalForce() every timestep.
# =============================================================================

import numpy as np
import pybullet as p


def _rotate_local_to_world(local_pos, body_pos, body_orn):
    """
    Rotate a point from a body's local frame to the world frame.

    Parameters
    ----------
    local_pos : array-like, shape (3,)
        Position in the body's local frame (meters).
    body_pos : array-like, shape (3,)
        Body's world-frame position from getBasePositionAndOrientation.
    body_orn : array-like, shape (4,)
        Body's world-frame quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray, shape (3,)
        World-frame position of the point.
    """
    # Build rotation matrix from quaternion
    rot_matrix = np.array(
        p.getMatrixFromQuaternion(body_orn)
    ).reshape(3, 3)
    world_pos = np.array(body_pos) + rot_matrix @ np.array(local_pos)
    return world_pos


class Connector:
    """
    A single breakable spring-damper connector between two rigid bodies.

    In the HDD context:
    - body_lid   : the HDD cover (moves during disassembly)
    - body_housing: the HDD base (fixed to world)
    - anchor_lid  : local position on the cover (e.g., screw hole location)
    - anchor_housing: corresponding local position on the housing

    The connector models a loosened-but-not-removed screw with residual
    clamping force, or a snap-fit clip. Both behave as spring-dampers
    until the separation force exceeds their strength.

    Parameters
    ----------
    physics_client : int
        PyBullet physics client ID (from p.connect()).
    body_lid : int
        PyBullet body ID of the HDD cover.
    body_housing : int
        PyBullet body ID of the HDD housing.
    anchor_lid_local : array-like, shape (3,)
        Connector anchor on the lid in lid's local frame (meters).
    anchor_housing_local : array-like, shape (3,)
        Connector anchor on the housing in housing's local frame (meters).
    k : float
        Spring stiffness (N/m). Typical range: 3000–8000 N/m.
    c : float
        Damping coefficient (N·s/m). Typical range: 20–80 N·s/m.
    strength : float
        Failure force threshold (N). When connector force >= strength,
        the connector fails and never applies force again.
    connector_id : int, optional
        An integer label for logging (e.g., 0–7 for HF connectors).
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
        self.anchor_lid_local = np.array(anchor_lid_local, dtype=float)
        self.anchor_housing_local = np.array(anchor_housing_local, dtype=float)
        self.k = float(k)
        self.c = float(c)
        self.strength = float(strength)
        self.connector_id = connector_id

        # State variables
        self.failed = False
        self.failure_time = None          # simulation time when failure occurred
        self.prev_extension_mag = 0.0     # for computing extension rate
        self.peak_force = 0.0             # track peak for logging

    def _get_world_anchors(self):
        """
        Compute current world-frame positions of both anchor points.
        Called every timestep because the lid moves.
        """
        lid_pos, lid_orn = p.getBasePositionAndOrientation(
            self.body_lid, physicsClientId=self.client
        )
        housing_pos, housing_orn = p.getBasePositionAndOrientation(
            self.body_housing, physicsClientId=self.client
        )
        world_anchor_lid = _rotate_local_to_world(
            self.anchor_lid_local, lid_pos, lid_orn
        )
        world_anchor_housing = _rotate_local_to_world(
            self.anchor_housing_local, housing_pos, housing_orn
        )
        return world_anchor_lid, world_anchor_housing

    def step(self, dt, current_time):
        """
        Compute and apply connector forces for one simulation timestep.

        This must be called every timestep BEFORE p.stepSimulation().

        Parameters
        ----------
        dt : float
            Simulation timestep (seconds).
        current_time : float
            Current simulation time (seconds), used for failure logging.

        Returns
        -------
        float
            Magnitude of the connector force this timestep (N).
            Returns 0.0 if the connector has already failed.
        """
        if self.failed:
            return 0.0

        # Get current world-frame anchor positions
        anchor_lid, anchor_housing = self._get_world_anchors()

        # Extension vector: points FROM housing anchor TO lid anchor.
        # When the lid is pulled up, this vector has a positive Z component.
        extension_vec = anchor_lid - anchor_housing
        extension_mag = np.linalg.norm(extension_vec)

        # Avoid division by zero when extension is negligible
        if extension_mag < 1e-9:
            self.prev_extension_mag = 0.0
            return 0.0

        unit_vec = extension_vec / extension_mag

        # Rate of extension change (finite difference)
        extension_rate = (extension_mag - self.prev_extension_mag) / dt

        # Spring-damper force magnitude
        force_mag = self.k * extension_mag + self.c * extension_rate

        # Clamp to zero: damper should not pull connector tighter than spring
        force_mag = max(force_mag, 0.0)

        # Check failure criterion
        if force_mag >= self.strength:
            self.failed = True
            self.failure_time = current_time
            self.prev_extension_mag = extension_mag
            # Do NOT apply force at failure timestep — connector snaps
            return force_mag  # return for logging before zeroing

        # Track peak force for later analysis
        self.peak_force = max(self.peak_force, force_mag)

        # Force vector restoring lid toward housing
        # Applied to lid: pulls lid DOWN (resists removal)
        force_on_lid = -force_mag * unit_vec
        # Applied to housing: equal and opposite (Newton's 3rd law)
        force_on_housing = force_mag * unit_vec

        # Apply forces at world-frame anchor positions.
        # Using WORLD_FRAME with the anchor position ensures the force
        # creates a torque if it is not aligned with the center of mass.
        p.applyExternalForce(
            self.body_lid,
            -1,  # -1 = base link
            force_on_lid.tolist(),
            anchor_lid.tolist(),
            p.WORLD_FRAME,
            physicsClientId=self.client,
        )
        p.applyExternalForce(
            self.body_housing,
            -1,
            force_on_housing.tolist(),
            anchor_housing.tolist(),
            p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        self.prev_extension_mag = extension_mag
        return force_mag

    def reset(self):
        """Reset connector to intact state. Used between simulation runs."""
        self.failed = False
        self.failure_time = None
        self.prev_extension_mag = 0.0
        self.peak_force = 0.0

    @property
    def is_intact(self):
        return not self.failed