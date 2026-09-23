from tasks.hdd_removal import HDDRemovalTask
from tasks.linear_extraction import LinearExtractionTask


def create_task(
    task_name,
    physics_client,
    fidelity,
    strength,
    friction,
    misalignment,
    gui=False,
    dt=1.0 / 240.0,
):
    if task_name == "hdd":
        return HDDRemovalTask(
            physics_client=physics_client,
            fidelity=fidelity,
            S=strength,
            mu=friction,
            delta=misalignment,
            gui=gui,
            dt=dt,
        )

    if task_name in {"pcb", "gpu"}:
        return LinearExtractionTask(
            physics_client=physics_client,
            task_name=task_name,
            fidelity=fidelity,
            strength=strength,
            friction=friction,
            misalignment=misalignment,
            gui=gui,
            dt=dt,
        )

    raise ValueError(
        f"Unknown task '{task_name}'. "
        "Expected hdd, pcb, or gpu."
    )