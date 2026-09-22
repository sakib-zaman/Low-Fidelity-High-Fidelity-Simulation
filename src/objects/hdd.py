# src/objects/hdd.py
import os
import numpy as np
import pybullet as p

# 2.5-inch Enterprise SAS drive standard dimensions
BASE_HALF   = [0.035, 0.050, 0.0075]   # 70mm x 100mm x 15mm
COVER_HALF  = [0.0345, 0.0495, 0.0015] # 69mm x 99mm x 3mm
COVER_MASS  = 0.065                    # 65 grams aluminum cover

# 8 Perimeter Attachment Screw Positions (relative to cover center)
HDD_ANCHORS = [
    [-0.030, -0.044, -COVER_HALF[2]],
    [ 0.000, -0.044, -COVER_HALF[2]],
    [ 0.030, -0.044, -COVER_HALF[2]],
    [-0.030,  0.000, -COVER_HALF[2]],
    [ 0.030,  0.000, -COVER_HALF[2]],
    [-0.030,  0.044, -COVER_HALF[2]],
    [ 0.000,  0.044, -COVER_HALF[2]],
    [ 0.030,  0.044, -COVER_HALF[2]],
]

HDD_STRENGTH_WEIGHTS = np.array([1.00, 0.95, 1.02, 0.97, 1.04, 1.01, 0.93, 1.08])
HDD_STRENGTH_WEIGHTS = HDD_STRENGTH_WEIGHTS / HDD_STRENGTH_WEIGHTS.sum() * 8.0

class HDDObject:
    def __init__(self, physics_client, position, mu=0.45, delta=0.0, fidelity='HF'):
        self.client   = physics_client
        self.pos      = list(position)
        self.mu       = mu
        self.delta    = delta
        self.fidelity = fidelity
        self.housing_id  = None
        self.cover_id    = None
        self.debug_lines = []
        self._build()

    def _build(self):
        base_mesh  = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "hdd_base.obj")
        cover_mesh = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "hdd_cover.obj")
        base_mesh  = os.path.abspath(base_mesh)
        cover_mesh = os.path.abspath(cover_mesh)

        # 1. Base Housing (Fixed to bench)
        base_z = self.pos[2] + BASE_HALF[2]
        col_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=BASE_HALF, physicsClientId=self.client)
        if os.path.exists(base_mesh):
            vis_base = p.createVisualShape(
                p.GEOM_MESH,
                fileName=base_mesh,
                rgbaColor=[0.25, 0.26, 0.28, 1.0],
                visualFramePosition=[0.0, 0.0, -BASE_HALF[2]],
                physicsClientId=self.client,
            )
        else:
            vis_base = p.createVisualShape(p.GEOM_BOX, halfExtents=BASE_HALF, rgbaColor=[0.25, 0.26, 0.28, 1.0], physicsClientId=self.client)

        self.housing_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=[self.pos[0], self.pos[1], base_z],
            physicsClientId=self.client
        )
        p.changeDynamics(self.housing_id, -1, lateralFriction=self.mu, physicsClientId=self.client)

        # 2. Cover Plate
        cover_z = self.pos[2] + (BASE_HALF[2] * 2.0) + COVER_HALF[2]
        cover_x = self.pos[0] + self.delta

        col_cover = p.createCollisionShape(p.GEOM_BOX, halfExtents=COVER_HALF, physicsClientId=self.client)
        if os.path.exists(cover_mesh):
            vis_cover = p.createVisualShape(
                p.GEOM_MESH,
                fileName=cover_mesh,
                rgbaColor=[0.85, 0.86, 0.89, 1.0],
                visualFramePosition=[0.0, 0.0, -COVER_HALF[2]],
                physicsClientId=self.client,
            )
        else:
            vis_cover = p.createVisualShape(p.GEOM_BOX, halfExtents=COVER_HALF, rgbaColor=[0.85, 0.86, 0.89, 1.0], physicsClientId=self.client)

        self.cover_id = p.createMultiBody(
            baseMass=COVER_MASS,
            baseCollisionShapeIndex=col_cover,
            baseVisualShapeIndex=vis_cover,
            basePosition=[cover_x, self.pos[1], cover_z],
            physicsClientId=self.client
        )
        p.changeDynamics(self.cover_id, -1, lateralFriction=self.mu, linearDamping=0.02, angularDamping=0.02, physicsClientId=self.client)

    def get_cover_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.cover_id, physicsClientId=self.client)
        return list(pos)

    def get_cover_orientation(self):
        _, orn = p.getBasePositionAndOrientation(self.cover_id, physicsClientId=self.client)
        return list(orn)

    def get_grasp_position(self):
        # Grasp right on top of the cover
        cpos = self.get_cover_position()
        return [cpos[0], cpos[1], cpos[2] + COVER_HALF[2] + 0.005]

    def get_pregrasp_position(self):
        gp = self.get_grasp_position()
        return [gp[0], gp[1], gp[2] + 0.12]

    def get_place_position(self):
        # Move 16cm to the right on the workbench surface
        cpos = self.get_cover_position()
        return [cpos[0] + 0.16, cpos[1], self.pos[2] + COVER_HALF[2] + 0.002]

    def draw_connectors(self, connectors):
        for line in self.debug_lines:
            p.removeUserDebugItem(line, physicsClientId=self.client)
        self.debug_lines = []

        cpos, corn = p.getBasePositionAndOrientation(self.cover_id, physicsClientId=self.client)
        hpos, horn = p.getBasePositionAndOrientation(self.housing_id, physicsClientId=self.client)
        Rc = np.array(p.getMatrixFromQuaternion(corn)).reshape(3, 3)
        Rh = np.array(p.getMatrixFromQuaternion(horn)).reshape(3, 3)

        for conn in connectors:
            wa = np.array(cpos) + Rc @ conn.anchor_lid_local
            wb = np.array(hpos) + Rh @ conn.anchor_housing_local
            color = [1.0, 0.1, 0.1] if conn.failed else [0.1, 0.9, 0.2]
            lid = p.addUserDebugLine(wa.tolist(), wb.tolist(), color, lineWidth=3, physicsClientId=self.client)
            self.debug_lines.append(lid)