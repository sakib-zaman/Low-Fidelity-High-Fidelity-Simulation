# src/objects/generate_hdd_mesh.py
import os
import numpy as np

def write_obj_box(vertices, faces, center, size, v_offset=1):
    dx, dy, dz = size[0]/2.0, size[1]/2.0, size[2]/2.0
    cx, cy, cz = center
    corners = [
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ]
    for c in corners:
        vertices.append(c)
    box_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 7, 6, 5],  # top
        [0, 4, 5, 1],  # front
        [1, 5, 6, 2],  # right
        [2, 6, 7, 3],  # back
        [3, 7, 4, 0],  # left
    ]
    for bf in box_faces:
        faces.append([i + v_offset for i in bf])
    return v_offset + 8

def write_obj_cylinder(vertices, faces, center, radius, height, segments=16, v_offset=1):
    cx, cy, cz = center
    half_h = height / 2.0
    # Bottom circle vertices
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        vertices.append([cx + radius * np.cos(theta), cy + radius * np.sin(theta), cz - half_h])
    # Top circle vertices
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        vertices.append([cx + radius * np.cos(theta), cy + radius * np.sin(theta), cz + half_h])
    # Center caps
    vertices.append([cx, cy, cz - half_h]) # bottom center
    vertices.append([cx, cy, cz + half_h]) # top center
    bot_center = v_offset + 2 * segments
    top_center = v_offset + 2 * segments + 1

    for i in range(segments):
        next_i = (i + 1) % segments
        # Bottom cap
        faces.append([bot_center, v_offset + i, v_offset + next_i])
        # Top cap
        faces.append([top_center, v_offset + segments + next_i, v_offset + segments + i])
        # Side quad
        faces.append([
            v_offset + i,
            v_offset + next_i,
            v_offset + segments + next_i,
            v_offset + segments + i
        ])
    return v_offset + 2 * segments + 2

def build_hdd_base_obj(filename="assets/hdd_base.obj"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    verts, faces = [], []
    vo = 1

    # Dimensions for 2.5" SAS Enterprise Drive: 70mm x 100mm x 15mm
    # Main cast aluminum chassis block
    vo = write_obj_box(verts, faces, [0, 0, 0.0075], [0.070, 0.100, 0.015], vo)
    # Side mounting rails (flanges)
    vo = write_obj_box(verts, faces, [-0.034, 0, 0.0075], [0.003, 0.096, 0.013], vo)
    vo = write_obj_box(verts, faces, [ 0.034, 0, 0.0075], [0.003, 0.096, 0.013], vo)
    # SATA / SAS Gold Connector Ports at rear edge
    vo = write_obj_box(verts, faces, [0.012, 0.048, 0.005], [0.025, 0.008, 0.006], vo)
    vo = write_obj_box(verts, faces, [-0.018, 0.048, 0.005], [0.016, 0.008, 0.006], vo)
    # Circular spindle motor hub in the center
    vo = write_obj_cylinder(verts, faces, [0, -0.010, 0.0152], radius=0.012, height=0.001, segments=20, v_offset=vo)

    with open(filename, "w") as f:
        f.write("# Procedural Enterprise 2.5-inch HDD Base Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx) for idx in face) + "\n")
    print(f"Generated {filename}")

def build_hdd_cover_obj(filename="assets/hdd_cover.obj"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    verts, faces = [], []
    vo = 1

    # Main stamped sheet metal plate: 69mm x 99mm x 2.5mm
    vo = write_obj_box(verts, faces, [0, 0, 0.00125], [0.069, 0.099, 0.0025], vo)
    # Stamped center reinforcement plate (raised surface)
    vo = write_obj_box(verts, faces, [0, -0.005, 0.0027], [0.058, 0.076, 0.0006], vo)
    # Recessed white label badge
    vo = write_obj_box(verts, faces, [0, 0.010, 0.0031], [0.050, 0.045, 0.0003], vo)
    # Spindle clamp disc boss
    vo = write_obj_cylinder(verts, faces, [0, -0.018, 0.0033], radius=0.010, height=0.0008, segments=20, v_offset=vo)
    # Breather hole filter dot
    vo = write_obj_cylinder(verts, faces, [0.022, -0.038, 0.0032], radius=0.003, height=0.0006, segments=12, v_offset=vo)
    # 8 Perimeter Torx Screw Pockets
    screw_coords = [
        [-0.030, -0.044], [0.0, -0.044], [0.030, -0.044],
        [-0.030,  0.000],                 [0.030,  0.000],
        [-0.030,  0.044], [0.0,  0.044], [0.030,  0.044]
    ]
    for sc in screw_coords:
        vo = write_obj_cylinder(verts, faces, [sc[0], sc[1], 0.0032], radius=0.0022, height=0.0008, segments=10, v_offset=vo)

    with open(filename, "w") as f:
        f.write("# Procedural Enterprise 2.5-inch HDD Cover Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx) for idx in face) + "\n")
    print(f"Generated {filename}")

if __name__ == "__main__":
    build_hdd_base_obj()
    build_hdd_cover_obj()