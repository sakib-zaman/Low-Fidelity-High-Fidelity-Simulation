# src/visualize.py
# =============================================================================
# GUI demonstration of Franka Panda HDD disassembly in a robotics lab.
#
# Usage examples:
#   python src/visualize.py --fidelity hf --strength 70 --friction 0.45 --misalignment-mm 4
#   python src/visualize.py --fidelity lf --strength 50 --friction 0.2 --misalignment-mm 0
#   python src/visualize.py --fidelity hf --strength 100 --friction 0.7 --misalignment-mm 6
# =============================================================================

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pybullet as p


def print_banner(fidelity, S, mu, delta_mm):
    print("\n" + "="*60)
    print("  MSEC 2027 — Multi-Fidelity HDD Disassembly Demo")
    print("="*60)
    print(f"  Fidelity:      {'HIGH (8 connectors)' if fidelity=='HF' else 'LOW (1 connector)'}")
    print(f"  Strength:      {S} N")
    print(f"  Friction:      {mu}")
    print(f"  Misalignment:  {delta_mm} mm")
    print("="*60)


def print_result(result):
    print("\n" + "="*60)
    print("  TASK RESULT")
    print("="*60)
    status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
    print(f"  Outcome:           {status}")
    print(f"  Peak force:        {result['peak_force']:.1f} N")
    print(f"  Max rotation:      {result['max_rotation_deg']:.1f}°")
    print(f"  Safety violated:   {result['safety_violated']}")
    print(f"  Connectors failed: {result['n_connectors_failed']}")
    print(f"  Wall-clock time:   {result['wall_time']:.1f} s")

    if result["failure_sequence"]:
        print("\n  Connector failure sequence:")
        for cid, t in result["failure_sequence"]:
            if t is not None:
                print(f"    Connector {cid}: t = {t:.4f} s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Franka Panda HDD cover removal."
    )
    parser.add_argument(
        "--fidelity", choices=["lf", "hf"], default="hf",
        help="Simulation fidelity: lf=1 connector, hf=8 connectors"
    )
    parser.add_argument(
        "--strength", type=float, default=70.0,
        help="Total attachment strength in Newtons (20-120)"
    )
    parser.add_argument(
        "--friction", type=float, default=0.45,
        help="Lid-housing friction coefficient (0.1-0.8)"
    )
    parser.add_argument(
        "--misalignment-mm", type=float, default=4.0,
        help="Initial cover misalignment in millimeters (0-8)"
    )
    args = parser.parse_args()

    fidelity  = args.fidelity.upper()
    delta_m   = args.misalignment_mm / 1000.0

    print_banner(fidelity, args.strength, args.friction,
                 args.misalignment_mm)

    # Connect GUI
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, 0,         # hide side panel for cleaner view
        physicsClientId=client
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_MOUSE_PICKING, 1,
        physicsClientId=client
    )

    # Import task here so sys.path is set first
    from tasks.hdd_removal import HDDRemovalTask

    task = HDDRemovalTask(
        physics_client=client,
        fidelity=fidelity,
        S=args.strength,
        mu=args.friction,
        delta=delta_m,
        gui=True,
        dt=1.0/120.0   # 120Hz — smooth GUI, stable physics
    )

    try:
        result = task.run()
        print_result(result)
        input("\n  Simulation complete. "
              "Press Enter to close the window...")
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()