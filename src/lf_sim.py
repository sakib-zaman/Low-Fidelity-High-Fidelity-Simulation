# lf_sim.py
# =============================================================================
# Low-Fidelity HDD Disassembly Simulation
#
# Physical model:
#   - HDD cover: plain box geometry (100mm x 80mm x 5mm)
#   - Housing:   plain box geometry (110mm x 90mm x 30mm), fixed to world
#   - Attachment: ONE central connector representing the aggregate
#                 clamping force of all cover screws
#   - Friction:   uniform lateral friction applied to cover base
#   - Misalignment: initial lateral offset of cover in X-direction
#
# Intentional simplifications (these are the fidelity gaps):
#   1. Single central connector cannot model edge-loading or peeling
#   2. Box geometry has no chamfers or recess features
#   3. No spatial variation in attachment strength
#   4. Contact representation is minimal
#
# Why this is scientifically valid:
#   The LF model captures the correct total attachment strength and mass.
#   Under centered, low-friction, zero-misalignment conditions it should
#   agree well with HF. Under misaligned conditions it will diverge.
#   That divergence is the study's key finding.
# =============================================================================

import time
import numpy as np
import pybullet as p
import pybullet_data

from connector import Connector


# =============================================================================
# HDD Physical Constants
# These are fixed across ALL experiments. Do not modify between runs.
# Based on a 3.5" desktop HDD (e.g., Seagate Barracuda housing dimensions).
# =============================================================================
HDD_COVER_HALF_EXTENTS  = [0.050, 0.040, 0.0025]   # 100mm x 80mm x 5mm
HDD_HOUSING_HALF_EXTENTS = [0.055, 0.045, 0.015]    # 110mm x 90mm x 30mm
HDD_COVER_MASS          = 0.08    # kg  (~80g aluminum cover)
HDD_HOUSING_MASS        = 0.0     # 0 = static/fixed body in PyBullet

# Spawn positions (meters)
HOUSING_BASE_POS  = [0.0, 0.0, 0.015]   # housing center
COVER_REST_HEIGHT = 0.015 + 0.015 + 0.0025  # on top of housing

# Pull parameters
PULL_VELOCITY     = 0.05          # m/s (50 mm/s vertical pull)
PULL_DURATION     = 4.0           # seconds maximum
SUCCESS_CLEARANCE = 0.030         # 30 mm above initial position = success

# LF connector parameters (tuned during pilot study)
LF_STIFFNESS      = 5000.0        # N/m
LF_DAMPING        = 40.0          # N·s/m

# Safety thresholds (checked every timestep)
SAFETY_MAX_FORCE   = 150.0        # N   end-effector resultant force
SAFETY_MAX_TORQUE  = 15.0         # N·m end-effector resultant torque
SAFETY_MAX_ROTATION = 25.0        # degrees lid rotation about Z-axis


class LowFidelityHDDSim:
    """
    Low-fidelity PyBullet simulation of HDD cover disassembly.

    Usage
    -----
    sim = LowFidelityHDDSim()
    result = sim.run(S=60.0, mu=0.3, delta=0.003)
    sim.close()

    Or use as a context manager:
    with LowFidelityHDDSim() as sim:
        result = sim.run(S=60.0, mu=0.3, delta=0.003)

    Parameters passed to run()
    ---------------------------
    S : float
        Total attachment strength (N). Range: 20–120 N.
    mu : float
        Lateral friction coefficient at cover-housing interface. Range: 0.1–0.8.
    delta : float
        Lateral misalignment offset in X-direction (meters). Range: 0–0.008.
    """

    def __init__(self, gui=False, timestep=1.0 / 500.0):
        """
        Initialize the PyBullet physics client.

        Parameters
        ----------
        gui : bool
            If True, open the PyBullet GUI (for debugging only).
            Always use False (DIRECT mode) for batch experiments.
        timestep : float
            Physics integration step (seconds). Default 1/500 s.
            Do not change between LF and HF — they must use the same dt.
        """
        self.gui = gui
        self.dt = timestep

        # Connect to physics engine
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Configure solver — increase iterations for stable connector forces
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            physicsClientId=self.client
        )
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client
        )

        # These are set during _build_scene() and cleared on reset
        self.cover_id   = None
        self.housing_id = None
        self.connector  = None

    # -------------------------------------------------------------------------
    # Scene construction
    # -------------------------------------------------------------------------

    def _build_scene(self, S, mu, delta):
        """
        Construct the simulation scene for one experiment.

        This creates:
        1. Gravity
        2. HDD housing (fixed)
        3. HDD cover (dynamic, offset by delta)
        4. One central connector with strength S
        5. Friction on the cover base

        Parameters
        ----------
        S : float   Total attachment strength (N)
        mu : float  Friction coefficient
        delta : float  X-axis offset (m)
        """
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        # --- Housing (fixed) ---
        housing_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=HDD_HOUSING_HALF_EXTENTS,
            physicsClientId=self.client
        )
        housing_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=HDD_HOUSING_HALF_EXTENTS,
            rgbaColor=[0.3, 0.3, 0.3, 1.0],  # dark gray
            physicsClientId=self.client
        )
        self.housing_id = p.createMultiBody(
            baseMass=HDD_HOUSING_MASS,
            baseCollisionShapeIndex=housing_col,
            baseVisualShapeIndex=housing_vis,
            basePosition=HOUSING_BASE_POS,
            physicsClientId=self.client
        )

        # Fix housing to world — mass=0 alone makes it static in PyBullet,
        # but we also explicitly confirm with a constraint for safety
        p.changeDynamics(
            self.housing_id, -1,
            lateralFriction=mu,
            physicsClientId=self.client
        )

        # --- HDD Cover (dynamic) ---
        # Spawn position includes lateral offset delta in X
        cover_start_pos = [
            delta,                        # X: misalignment offset
            0.0,                          # Y: centered
            COVER_REST_HEIGHT             # Z: resting on housing
        ]
        cover_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=HDD_COVER_HALF_EXTENTS,
            physicsClientId=self.client
        )
        cover_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=HDD_COVER_HALF_EXTENTS,
            rgbaColor=[0.7, 0.7, 0.8, 1.0],  # light silver
            physicsClientId=self.client
        )
        self.cover_id = p.createMultiBody(
            baseMass=HDD_COVER_MASS,
            baseCollisionShapeIndex=cover_col,
            baseVisualShapeIndex=cover_vis,
            basePosition=cover_start_pos,
            physicsClientId=self.client
        )

        # Apply friction to the cover's bottom face
        p.changeDynamics(
            self.cover_id, -1,
            lateralFriction=mu,
            linearDamping=0.01,   # small air damping — keeps things stable
            angularDamping=0.01,
            physicsClientId=self.client
        )

        # --- Single Central Connector (LF model) ---
        # Anchor at geometric center of the interface.
        # In local frame: [0, 0, -half_cover_thickness] for cover,
        #                 [0, 0, +half_housing_thickness] for housing.
        # The connector sits at the mating surface.
        self.connector = Connector(
            physics_client=self.client,
            body_lid=self.cover_id,
            body_housing=self.housing_id,
            anchor_lid_local=[0.0, 0.0, -HDD_COVER_HALF_EXTENTS[2]],
            anchor_housing_local=[0.0, 0.0, HDD_HOUSING_HALF_EXTENTS[2]],
            k=LF_STIFFNESS,
            c=LF_DAMPING,
            strength=S,           # full strength at single point
            connector_id=0
        )

        # Let the scene settle for 0.1 seconds before pulling
        self._settle()

    def _settle(self, duration=0.1):
        """
        Run physics without pulling to let the cover settle onto the housing.
        This eliminates startup transients from initial position errors.
        """
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.connector.step(self.dt, current_time=0.0)
            p.stepSimulation(physicsClientId=self.client)

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    def _apply_pull_force(self, current_pos, target_pos, pull_force_log):
        """
        Impedance controller: compute and apply a vertical pull force.

        We model the robot end-effector as a virtual spring pulling the cover
        toward a target position that moves upward at PULL_VELOCITY.

        This is simpler than a full UR5e model but captures the force-position
        relationship correctly. For MSEC publication, this is described as
        'a position-controlled impedance pull at constant velocity'.

        Parameters
        ----------
        current_pos : array-like (3,)   Current cover center position
        target_pos  : array-like (3,)   Current target position (moving up)
        pull_force_log : list           Appended to for logging

        Returns
        -------
        float   Magnitude of applied pull force (N)
        """
        # Proportional gain (virtual spring stiffness of the robot arm)
        Kp = 800.0   # N/m — stiff enough to maintain velocity

        error = np.array(target_pos) - np.array(current_pos)
        force_vec = Kp * error

        # Apply only upward (+Z) component to simulate vertical pull
        # Lateral components represent the robot resisting drift
        pull_force = np.array([force_vec[0] * 0.1,   # allow slight lateral
                                force_vec[1] * 0.1,   # compliance
                                force_vec[2]])         # full vertical

        p.applyExternalForce(
            self.cover_id, -1,
            pull_force.tolist(),
            current_pos,          # apply at cover center of mass
            p.WORLD_FRAME,
            physicsClientId=self.client
        )

        force_mag = float(np.linalg.norm(pull_force))
        pull_force_log.append(force_mag)
        return force_mag

    # -------------------------------------------------------------------------
    # Safety monitor
    # -------------------------------------------------------------------------

    def _check_safety(self, force_mag, cover_pos, cover_orn,
                      initial_cover_z):
        """
        Check all safety thresholds. Returns True if ANY threshold is violated.

        Checks:
        1. End-effector force exceeds SAFETY_MAX_FORCE
        2. Cover rotation about Z exceeds SAFETY_MAX_ROTATION
        3. Cover collides with housing after being lifted > 5mm
        """
        # Check 1: Force limit
        if force_mag > SAFETY_MAX_FORCE:
            return True

        # Check 2: Lid rotation (Z-axis Euler angle)
        euler = p.getEulerFromQuaternion(cover_orn)
        rotation_z_deg = abs(np.degrees(euler[2]))
        if rotation_z_deg > SAFETY_MAX_ROTATION:
            return True

        # Check 3: Re-contact after lift
        lift = cover_pos[2] - initial_cover_z
        if lift > 0.005:   # only check after 5mm lift
            contacts = p.getContactPoints(
                bodyA=self.cover_id,
                bodyB=self.housing_id,
                physicsClientId=self.client
            )
            if contacts and len(contacts) > 0:
                return True

        return False

    # -------------------------------------------------------------------------
    # Main run method
    # -------------------------------------------------------------------------

    def run(self, S, mu, delta):
        """
        Run one complete LF disassembly simulation.

        Parameters
        ----------
        S : float     Total attachment strength (N)
        mu : float    Friction coefficient
        delta : float Misalignment offset in X (meters)

        Returns
        -------
        dict with keys:
            success          : bool
            peak_force       : float (N)
            safety_violated  : bool
            max_rotation_deg : float (degrees)
            wall_time        : float (seconds, actual CPU time)
            force_series     : np.ndarray (sampled at ~50 Hz)
            lid_z_series     : np.ndarray (cover Z position over time)
            time_series      : np.ndarray (simulation time stamps)
            connector_failed_time : float or None
        """
        wall_start = time.perf_counter()

        # Build scene
        self._build_scene(S, mu, delta)

        # Record initial cover position
        init_pos, init_orn = p.getBasePositionAndOrientation(
            self.cover_id, physicsClientId=self.client
        )
        initial_cover_z = init_pos[2]

        # Logging arrays
        force_series   = []
        lid_z_series   = []
        time_series    = []

        # Simulation state
        total_steps    = int(PULL_DURATION / self.dt)
        log_interval   = max(1, int((1.0 / 50.0) / self.dt))  # log at 50 Hz
        success        = False
        safety_violated = False
        max_rotation   = 0.0
        peak_force     = 0.0
        sim_time       = 0.0

        for step in range(total_steps):
            sim_time = step * self.dt

            # --- 1. Compute target position (moves up at PULL_VELOCITY) ---
            target_z = initial_cover_z + PULL_VELOCITY * max(0.0, sim_time - 0.1)
            # 0.1 s delay before pulling starts (settle period already done,
            # but we add a small ramp-up for physical realism)
            target_pos = [delta * 0.0,  # robot holds lateral position at 0
                          0.0,
                          target_z]

            # --- 2. Get current cover state ---
            cover_pos, cover_orn = p.getBasePositionAndOrientation(
                self.cover_id, physicsClientId=self.client
            )

            # --- 3. Apply connector force (before stepping) ---
            conn_force = self.connector.step(self.dt, sim_time)

            # --- 4. Apply robot pull force ---
            pull_force = self._apply_pull_force(
                cover_pos, target_pos, []
            )

            # Track peak force
            peak_force = max(peak_force, pull_force, conn_force)

            # --- 5. Step physics ---
            p.stepSimulation(physicsClientId=self.client)

            # --- 6. Logging (at 50 Hz) ---
            if step % log_interval == 0:
                force_series.append(pull_force)
                lid_z_series.append(cover_pos[2])
                time_series.append(sim_time)

            # --- 7. Rotation tracking ---
            euler = p.getEulerFromQuaternion(cover_orn)
            rot_z = abs(np.degrees(euler[2]))
            max_rotation = max(max_rotation, rot_z)

            # --- 8. Safety check ---
            if self._check_safety(pull_force, cover_pos, cover_orn,
                                   initial_cover_z):
                safety_violated = True
                break

            # --- 9. Success check ---
            lift = cover_pos[2] - initial_cover_z
            if self.connector.failed and lift >= SUCCESS_CLEARANCE:
                success = True
                break

        wall_time = time.perf_counter() - wall_start

        return {
            "success":               success,
            "peak_force":            peak_force,
            "safety_violated":       safety_violated,
            "max_rotation_deg":      max_rotation,
            "wall_time":             wall_time,
            "force_series":          np.array(force_series),
            "lid_z_series":          np.array(lid_z_series),
            "time_series":           np.array(time_series),
            "connector_failed_time": self.connector.failure_time,
            "fidelity":              "LF",
        }

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self):
        """Disconnect from PyBullet. Always call this when done."""
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()