from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str

    object_size: Tuple[float, float, float]
    object_mass: float
    object_color: Tuple[float, float, float, float]

    fixture_size: Tuple[float, float, float]
    fixture_color: Tuple[float, float, float, float]

    object_position: Tuple[float, float, float]

    extraction_direction: Tuple[float, float, float]
    extraction_distance: float
    extraction_speed: float

    lf_attachment_count: int
    hf_attachment_count: int

    grasp_width: float
    pregrasp_offset: Tuple[float, float, float]
    grasp_offset: Tuple[float, float, float]
    placement_position: Tuple[float, float, float]


HDD_SPEC = TaskSpec(
    task_id=0,
    name="hdd",
    object_size=(0.069, 0.099, 0.003),
    object_mass=0.065,
    object_color=(0.82, 0.84, 0.88, 1.0),
    fixture_size=(0.070, 0.100, 0.015),
    fixture_color=(0.22, 0.23, 0.25, 1.0),
    object_position=(0.52, 0.0, 0.6965),
    extraction_direction=(0.0, 0.0, 1.0),
    extraction_distance=0.070,
    extraction_speed=0.035,
    lf_attachment_count=1,
    hf_attachment_count=8,
    grasp_width=0.068,
    pregrasp_offset=(0.0, 0.0, 0.120),
    grasp_offset=(0.0, 0.0, 0.008),
    placement_position=(0.72, -0.18, 0.6845),
)


PCB_SPEC = TaskSpec(
    task_id=1,
    name="pcb",
    object_size=(0.120, 0.002, 0.080),
    object_mass=0.090,
    object_color=(0.05, 0.38, 0.12, 1.0),
    fixture_size=(0.145, 0.060, 0.110),
    fixture_color=(0.25, 0.27, 0.30, 1.0),
    object_position=(0.52, 0.0, 0.750),
    extraction_direction=(0.0, -1.0, 0.0),
    extraction_distance=0.120,
    extraction_speed=0.025,
    lf_attachment_count=1,
    hf_attachment_count=12,
    grasp_width=0.030,
    pregrasp_offset=(0.0, -0.100, 0.030),
    grasp_offset=(0.0, -0.012, 0.0),
    placement_position=(0.72, -0.20, 0.684),
)


GPU_SPEC = TaskSpec(
    task_id=2,
    name="gpu",
    object_size=(0.180, 0.012, 0.100),
    object_mass=0.750,
    object_color=(0.08, 0.12, 0.10, 1.0),
    fixture_size=(0.210, 0.070, 0.130),
    fixture_color=(0.20, 0.22, 0.25, 1.0),
    object_position=(0.50, 0.0, 0.760),
    extraction_direction=(0.0, -1.0, 0.0),
    extraction_distance=0.140,
    extraction_speed=0.020,
    lf_attachment_count=1,
    hf_attachment_count=10,
    grasp_width=0.060,
    pregrasp_offset=(0.0, -0.120, 0.050),
    grasp_offset=(0.0, -0.018, 0.015),
    placement_position=(0.70, -0.20, 0.690),
)


TASK_SPECS = {
    "hdd": HDD_SPEC,
    "pcb": PCB_SPEC,
    "gpu": GPU_SPEC,
}