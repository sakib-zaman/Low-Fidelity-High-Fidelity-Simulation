# src/simulation/scene.py
# =============================================================================
# Robotics lab environment.
#
# Builds a realistic lab scene containing:
#   - Tiled floor
#   - Three walls (back, left, right) with light grey paint
#   - Ceiling with recessed light panels
#   - Main steel workbench with lower shelf
#   - Tool pegboard on back wall
#   - Two equipment racks (grey server-style boxes)
#   - Overhead warning light strip (orange)
#   - Robot mounting base plate (heavy steel)
# =============================================================================

import pybullet as p
import pybullet_data
import numpy as np

# Lab dimensions
LAB_W  = 4.0    # width  (X)
LAB_D  = 3.5    # depth  (Y)
LAB_H  = 2.8    # height (Z)

# Workbench
BENCH_W = 1.4
BENCH_D = 0.7
BENCH_H = 0.63
BENCH_X = 0.55   # center X
BENCH_Y = 0.0    # center Y

TABLETOP_HALF_THICKNESS = 0.025
WORK_SURFACE_Z = BENCH_H + 2.0 * TABLETOP_HALF_THICKNESS


class Scene:
    """
    Full robotics disassembly lab environment.

    Parameters
    ----------
    physics_client : int
    gui            : bool
    """

    def __init__(self, physics_client, gui=False):
        self.client = physics_client
        self.gui    = gui
        self._build()

    def _box(self, half, pos, color, mass=0):
        """Helper: create a static colored box."""
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half,
            physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half,
            rgbaColor=color,
            physicsClientId=self.client
        )
        bid = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=self.client
        )
        return bid

    def _build(self):
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self._build_floor()
        self._build_walls()
        self._build_ceiling()
        self._build_workbench()
        self._build_robot_base()
        self._build_equipment_racks()
        self._build_pegboard()
        self._build_overhead_light()

        if self.gui:
            self._configure_camera()

    # ------------------------------------------------------------------

    def _build_floor(self):
        """Tiled floor — light grey with slight texture approximation."""
        # Base floor slab
        self._box(
            [LAB_W/2, LAB_D/2, 0.02],
            [0.0, 0.0, -0.02],
            [0.75, 0.75, 0.75, 1.0]   # medium grey tile
        )
        # Darker grout lines approximated with thin strips (X direction)
        for i in range(-2, 3):
            self._box(
                [LAB_W/2, 0.005, 0.001],
                [0.0, i * 0.6, 0.001],
                [0.50, 0.50, 0.50, 1.0]
            )
        # Grout lines (Y direction)
        for i in range(-3, 4):
            self._box(
                [0.005, LAB_D/2, 0.001],
                [i * 0.6, 0.0, 0.001],
                [0.50, 0.50, 0.50, 1.0]
            )

    def _build_walls(self):
        """Three walls: back (+Y), left (-X), right (+X)."""
        wall_t = 0.05
        color  = [0.88, 0.88, 0.86, 1.0]   # off-white lab wall

        # Back wall (+Y side)
        self._box(
            [LAB_W/2, wall_t/2, LAB_H/2],
            [0.0, LAB_D/2, LAB_H/2],
            color
        )
        # Left wall (-X side)
        self._box(
            [wall_t/2, LAB_D/2, LAB_H/2],
            [-LAB_W/2, 0.0, LAB_H/2],
            color
        )
        # Right wall (+X side)
        self._box(
            [wall_t/2, LAB_D/2, LAB_H/2],
            [LAB_W/2, 0.0, LAB_H/2],
            color
        )
        # Front wall (partial — only bottom half so camera can see in)
        self._box(
            [LAB_W/2, wall_t/2, LAB_H/4],
            [0.0, -LAB_D/2, LAB_H/4],
            color
        )

        # Baseboard trim (darker strip at floor level)
        trim_color = [0.55, 0.55, 0.53, 1.0]
        self._box([LAB_W/2, 0.01, 0.05],
                   [0.0, LAB_D/2 - 0.01, 0.05], trim_color)
        self._box([0.01, LAB_D/2, 0.05],
                   [-LAB_W/2 + 0.01, 0.0, 0.05], trim_color)
        self._box([0.01, LAB_D/2, 0.05],
                   [LAB_W/2 - 0.01, 0.0, 0.05], trim_color)

    def _build_ceiling(self):
        """White ceiling with recessed LED panels."""
        # Ceiling slab
        self._box(
            [LAB_W/2, LAB_D/2, 0.03],
            [0.0, 0.0, LAB_H + 0.03],
            [0.95, 0.95, 0.95, 1.0]
        )
        # LED light panels (bright white rectangles)
        panel_color = [1.0, 1.0, 0.97, 1.0]
        for py in [-0.8, 0.8]:
            self._box(
                [0.5, 0.15, 0.01],
                [0.2, py, LAB_H - 0.01],
                panel_color
            )

    def _build_workbench(self):
        """
        Steel workbench with:
          - Thick tabletop
          - Four legs
          - Lower shelf
          - Front edge strip
        """
        top_color  = [0.55, 0.58, 0.60, 1.0]   # steel grey
        leg_color  = [0.40, 0.42, 0.44, 1.0]
        shelf_color = [0.50, 0.52, 0.54, 1.0]

        # Tabletop
        self._box(
            [BENCH_W/2, BENCH_D/2, 0.025],
            [BENCH_X, BENCH_Y, BENCH_H + 0.025],
            top_color
        )
        # Front edge strip (slightly darker)
        self._box(
            [BENCH_W/2, 0.008, 0.030],
            [BENCH_X, BENCH_Y - BENCH_D/2, BENCH_H + 0.008],
            [0.35, 0.37, 0.39, 1.0]
        )

        # Four legs
        leg_h = BENCH_H / 2
        leg_r = 0.018
        for lx in [BENCH_X - BENCH_W/2 + 0.03,
                   BENCH_X + BENCH_W/2 - 0.03]:
            for ly in [BENCH_Y - BENCH_D/2 + 0.03,
                       BENCH_Y + BENCH_D/2 - 0.03]:
                self._box(
                    [leg_r, leg_r, leg_h],
                    [lx, ly, leg_h],
                    leg_color
                )

        # Lower shelf (at knee height)
        self._box(
            [BENCH_W/2 - 0.04, BENCH_D/2 - 0.04, 0.012],
            [BENCH_X, BENCH_Y, 0.35],
            shelf_color
        )

        # Shelf items: small tool boxes
        self._box([0.06, 0.04, 0.03],
                   [BENCH_X - 0.2, BENCH_Y - 0.1, 0.35 + 0.012 + 0.03],
                   [0.8, 0.3, 0.1, 1.0])   # red tool box
        self._box([0.04, 0.04, 0.025],
                   [BENCH_X + 0.1, BENCH_Y, 0.35 + 0.012 + 0.025],
                   [0.2, 0.4, 0.8, 1.0])   # blue container

    def _build_robot_base(self):
        """
        Heavy steel mounting base plate for the robot.
        Bolted to the bench surface left-center area.
        """
        self._box(
            [0.12, 0.12, 0.025],
            [0.0, 0.0, BENCH_H + 0.05 + 0.025],
            [0.30, 0.30, 0.32, 1.0]   # very dark steel
        )
        # Bolt details (four tiny boxes at corners)
        for bx in [-0.09, 0.09]:
            for by in [-0.09, 0.09]:
                self._box(
                    [0.006, 0.006, 0.008],
                    [bx, by, BENCH_H + 0.05 + 0.055],
                    [0.6, 0.6, 0.1, 1.0]   # yellow bolt heads
                )

    def _build_equipment_racks(self):
        """
        Two equipment racks against the back wall —
        simulate server chassis or test equipment.
        """
        rack_color  = [0.25, 0.27, 0.30, 1.0]
        panel_color = [0.15, 0.15, 0.15, 1.0]
        led_color   = [0.1, 0.9, 0.3, 1.0]

        for rx in [-1.2, 1.4]:
            # Main rack body
            self._box(
                [0.22, 0.35, 0.60],
                [rx, LAB_D/2 - 0.38, 0.60],
                rack_color
            )
            # Front panel
            self._box(
                [0.20, 0.01, 0.58],
                [rx, LAB_D/2 - 0.04, 0.60],
                panel_color
            )
            # Panel LED strips
            for lz in [0.25, 0.50, 0.75, 1.00]:
                self._box(
                    [0.15, 0.005, 0.004],
                    [rx, LAB_D/2 - 0.035, lz],
                    led_color
                )
            # Rack handle
            self._box(
                [0.06, 0.015, 0.012],
                [rx, LAB_D/2 - 0.025, 0.30],
                [0.7, 0.7, 0.7, 1.0]
            )

    def _build_pegboard(self):
        """
        Pegboard tool organizer on the back wall above the workbench.
        """
        board_color = [0.65, 0.58, 0.48, 1.0]   # wood/tan
        peg_color   = [0.30, 0.30, 0.30, 1.0]

        # Board
        self._box(
            [0.45, 0.015, 0.35],
            [BENCH_X, LAB_D/2 - 0.015, BENCH_H + 0.60],
            board_color
        )
        # Tool silhouettes (flat colored boxes suggesting hanging tools)
        tools = [
            ([0.05, 0.005, 0.12], [BENCH_X - 0.28, LAB_D/2 - 0.01, BENCH_H + 0.72],
             [0.20, 0.20, 0.20, 1.0]),   # wrench
            ([0.04, 0.005, 0.10], [BENCH_X - 0.10, LAB_D/2 - 0.01, BENCH_H + 0.70],
             [0.80, 0.30, 0.10, 1.0]),   # screwdriver
            ([0.06, 0.005, 0.08], [BENCH_X + 0.15, LAB_D/2 - 0.01, BENCH_H + 0.68],
             [0.20, 0.20, 0.20, 1.0]),   # pliers
        ]
        for half, pos, col in tools:
            self._box(half, pos, col)

        # Peg holes (small dark dots)
        for px in np.linspace(BENCH_X - 0.38, BENCH_X + 0.38, 8):
            for pz in np.linspace(BENCH_H + 0.40, BENCH_H + 0.85, 5):
                self._box(
                    [0.003, 0.008, 0.003],
                    [px, LAB_D/2 - 0.02, pz],
                    [0.20, 0.18, 0.15, 1.0]
                )

    def _build_overhead_light(self):
        """Orange warning light strip above robot workspace."""
        self._box(
            [0.20, 0.015, 0.010],
            [0.0, -0.3, LAB_H - 0.10],
            [0.95, 0.50, 0.05, 1.0]
        )

    def _configure_camera(self):
        """Set camera to a good front-right view of robot and HDD."""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=35,
            cameraPitch=-22,
            cameraTargetPosition=[0.45, 0.0, 0.70],
            physicsClientId=self.client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1,
            physicsClientId=self.client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
            physicsClientId=self.client
        )

    @property
    def work_surface_z(self):
        return WORK_SURFACE_Z