import argparse
import os
import time

import numpy as np
import pandas as pd
import pybullet as p
from scipy.stats.qmc import LatinHypercube, scale

from tasks.factory import create_task


TASKS = ["hdd", "pcb", "gpu"]


def run_once(
    task_name,
    fidelity,
    strength,
    friction,
    misalignment,
):
    client = p.connect(p.DIRECT)

    try:
        task = create_task(
            task_name=task_name,
            physics_client=client,
            fidelity=fidelity,
            strength=strength,
            friction=friction,
            misalignment=misalignment,
            gui=False,
            dt=1.0 / 240.0,
        )
        return task.run()
    finally:
        p.disconnect(client)


def adequacy_label(lf, hf):
    success_agreement = lf["success"] == hf["success"]
    safety_agreement = (
        lf["safety_violated"]
        == hf["safety_violated"]
    )
    placement_agreement = (
        lf["placement_success"]
        == hf["placement_success"]
    )

    force_error = abs(
        lf["peak_force"] - hf["peak_force"]
    ) / max(hf["peak_force"], 10.0)

    adequate = (
        success_agreement
        and safety_agreement
        and placement_agreement
        and force_error <= 0.10
    )

    return int(adequate), force_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-task", type=int, default=300)
    parser.add_argument(
        "--output",
        default="data/multitask_dataset.csv",
    )
    args = parser.parse_args()

    rows = []
    case_id = 0

    for task_id, task_name in enumerate(TASKS):
        sampler = LatinHypercube(
            d=3,
            seed=42 + task_id,
        )
        unit_samples = sampler.random(
            args.samples_per_task
        )

        samples = scale(
            unit_samples,
            [20.0, 0.10, 0.0],
            [120.0, 0.80, 0.008],
        )

        for strength, friction, misalignment in samples:
            print(
                f"Case {case_id}: {task_name}, "
                f"S={strength:.1f}, mu={friction:.2f}, "
                f"delta={misalignment * 1000:.2f}mm"
            )

            lf = run_once(
                task_name,
                "LF",
                strength,
                friction,
                misalignment,
            )
            hf = run_once(
                task_name,
                "HF",
                strength,
                friction,
                misalignment,
            )

            adequate, force_error = adequacy_label(
                lf,
                hf,
            )

            rows.append({
                "case_id": case_id,
                "task": task_name,
                "task_id": task_id,
                "strength": strength,
                "friction": friction,
                "misalignment": misalignment,
                "lf_success": int(lf["success"]),
                "hf_success": int(hf["success"]),
                "lf_placement": int(
                    lf["placement_success"]
                ),
                "hf_placement": int(
                    hf["placement_success"]
                ),
                "lf_peak_force": lf["peak_force"],
                "hf_peak_force": hf["peak_force"],
                "lf_wall_time": lf["wall_time"],
                "hf_wall_time": hf["wall_time"],
                "force_error": force_error,
                "adequate": adequate,
            })

            case_id += 1

    output_directory = os.path.dirname(
        args.output
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    pd.DataFrame(rows).to_csv(
        args.output,
        index=False,
    )

    print(f"Saved {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()