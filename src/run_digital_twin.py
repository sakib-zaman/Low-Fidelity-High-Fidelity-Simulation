import argparse
import json
import os
import time

import joblib
import pandas as pd
import pybullet as p

from tasks.factory import create_task


MODEL_DIR = "data/multitask_model"


def load_selector():
    model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "selector.joblib",
        )
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "metadata.json",
        )
    ) as metadata_file:
        metadata = json.load(metadata_file)

    return model, float(metadata["threshold"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AI-guided multi-fidelity robotic "
            "disassembly digital twin"
        )
    )
    parser.add_argument(
        "--task",
        choices=["hdd", "pcb", "gpu"],
        required=True,
    )
    parser.add_argument(
        "--strength",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--friction",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--misalignment-mm",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--gui",
        action="store_true",
    )
    args = parser.parse_args()

    model, threshold = load_selector()

    condition = pd.DataFrame([{
        "task": args.task,
        "strength": args.strength,
        "friction": args.friction,
        "misalignment": (
            args.misalignment_mm / 1000.0
        ),
    }])

    adequacy_probability = float(
        model.predict_proba(condition)[0, 1]
    )

    selected_fidelity = (
        "LF"
        if adequacy_probability >= threshold
        else "HF"
    )

    print("=" * 60)
    print("AI-GUIDED DIGITAL TWIN")
    print("=" * 60)
    print(f"Task:                    {args.task}")
    print(f"P(LF adequate):          {adequacy_probability:.3f}")
    print(f"Safety threshold:        {threshold:.3f}")
    print(f"AI-selected fidelity:    {selected_fidelity}")
    print("=" * 60)

    connection_mode = (
        p.GUI if args.gui else p.DIRECT
    )
    client = p.connect(connection_mode)

    try:
        task = create_task(
            task_name=args.task,
            physics_client=client,
            fidelity=selected_fidelity,
            strength=args.strength,
            friction=args.friction,
            misalignment=(
                args.misalignment_mm / 1000.0
            ),
            gui=args.gui,
            dt=1.0 / 240.0,
        )

        result = task.run()

        print("\nRESULT")
        print("-" * 60)

        for key, value in result.items():
            if key not in {
                "force_series",
                "time_series",
                "lid_z_series",
            }:
                print(f"{key:25s}: {value}")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()