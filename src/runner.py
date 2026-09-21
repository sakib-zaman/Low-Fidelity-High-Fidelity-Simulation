# runner.py
# =============================================================================
# Unified simulation runner and dataset manager.
#
# This module is the single entry point for all experiment scripts.
# It handles:
#   1. Running LF or HF simulation for a given condition
#   2. Computing the adequacy label from paired results
#   3. Saving/loading datasets as CSV and HDF5
#   4. Parallel batch execution
#
# The adequacy label is computed here — in one place — so it is never
# accidentally computed differently in different scripts.
# =============================================================================

import os
import time
import json
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from lf_sim import LowFidelityHDDSim
from hf_sim import HighFidelityHDDSim


# =============================================================================
# Adequacy label definition
# Defined as a module-level constant so it cannot accidentally vary.
# Primary threshold: 10% force error.
# =============================================================================
ADEQUACY_FORCE_THRESHOLD = 0.10   # 10% normalized force error
EPSILON_FORCE             = 1.0   # N, prevents division by zero


def compute_adequacy_label(lf_result, hf_result,
                            force_threshold=ADEQUACY_FORCE_THRESHOLD):
    """
    Compute whether the LF simulation was adequate for a given condition.

    LF is adequate (label = 1) if ALL three conditions hold:
      (i)  Success agreement:  lf_success == hf_success
      (ii) Safety agreement:   lf_safety_violated == hf_safety_violated
      (iii) Force agreement:   normalized force error <= force_threshold

    Parameters
    ----------
    lf_result : dict    Output from LowFidelityHDDSim.run()
    hf_result : dict    Output from HighFidelityHDDSim.run()
    force_threshold : float  Maximum acceptable normalized force error

    Returns
    -------
    int (0 or 1)   1 = LF adequate, 0 = LF inadequate
    float          Normalized force discrepancy
    """
    # Condition (i): success agreement
    success_agree = (lf_result["success"] == hf_result["success"])

    # Condition (ii): safety agreement
    safety_agree = (lf_result["safety_violated"] == hf_result["safety_violated"])

    # Condition (iii): normalized force error
    f_lf = lf_result["peak_force"]
    f_hf = hf_result["peak_force"]
    force_discrepancy = abs(f_lf - f_hf) / max(f_hf, EPSILON_FORCE)
    force_agree = (force_discrepancy <= force_threshold)

    adequate = int(success_agree and safety_agree and force_agree)
    return adequate, force_discrepancy


def run_paired(case_id, S, mu, delta, seed=0, force_threshold=ADEQUACY_FORCE_THRESHOLD):
    """
    Run both LF and HF simulations for a single condition and return
    a flat record suitable for a DataFrame row.

    Parameters
    ----------
    case_id : int     Unique case identifier
    S : float         Attachment strength (N)
    mu : float        Friction coefficient
    delta : float     Misalignment (m)
    seed : int        Random seed (for future stochastic extensions)
    force_threshold : float  Adequacy force threshold

    Returns
    -------
    dict   Flat record with all LF, HF, and derived fields
    """
    np.random.seed(seed)

    # Run LF
    with LowFidelityHDDSim(gui=False) as lf:
        lf_result = lf.run(S=S, mu=mu, delta=delta)

    # Run HF
    with HighFidelityHDDSim(gui=False) as hf:
        hf_result = hf.run(S=S, mu=mu, delta=delta)

    # Compute adequacy
    adequate_label, force_discrepancy = compute_adequacy_label(
        lf_result, hf_result, force_threshold
    )

    record = {
        # Metadata
        "case_id":              case_id,
        "seed":                 seed,
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),

        # Inputs
        "S":                    S,
        "mu":                   mu,
        "delta":                delta,

        # LF outputs
        "lf_success":           int(lf_result["success"]),
        "lf_peak_force":        lf_result["peak_force"],
        "lf_safety_violated":   int(lf_result["safety_violated"]),
        "lf_max_rotation_deg":  lf_result["max_rotation_deg"],
        "lf_wall_time":         lf_result["wall_time"],
        "lf_connector_failed_t":lf_result["connector_failed_time"],

        # HF outputs
        "hf_success":           int(hf_result["success"]),
        "hf_peak_force":        hf_result["peak_force"],
        "hf_safety_violated":   int(hf_result["safety_violated"]),
        "hf_max_rotation_deg":  hf_result["max_rotation_deg"],
        "hf_wall_time":         hf_result["wall_time"],
        "hf_connector_failed_t":hf_result["connector_failed_time"],
        "hf_failure_sequence":  json.dumps(hf_result["failure_sequence"]),
        "hf_n_failed":          sum(1 for _, t in hf_result["failure_sequence"]
                                    if t is not None),

        # Derived metrics
        "force_discrepancy":    force_discrepancy,
        "adequate_label":       adequate_label,
        "success_agree":        int(lf_result["success"] == hf_result["success"]),
        "safety_agree":         int(lf_result["safety_violated"] ==
                                    hf_result["safety_violated"]),
        "speedup_ratio":        (hf_result["wall_time"] /
                                 max(lf_result["wall_time"], 1e-6)),
    }

    return record, lf_result, hf_result


def save_time_series(output_path, case_id, lf_result, hf_result):
    """
    Append force and position time series to an HDF5 file.
    Each case gets its own group keyed by case_id.

    HDF5 is used because CSV cannot efficiently store variable-length arrays.
    """
    with h5py.File(output_path, "a") as f:
        grp = f.require_group(f"case_{case_id:05d}")
        for key in ["force_series", "lid_z_series", "time_series"]:
            if key in lf_result and lf_result[key] is not None:
                grp.create_dataset(f"lf_{key}", data=lf_result[key],
                                   compression="gzip")
            if key in hf_result and hf_result[key] is not None:
                grp.create_dataset(f"hf_{key}", data=hf_result[key],
                                   compression="gzip")
        if "active_connectors" in hf_result:
            grp.create_dataset("hf_active_connectors",
                                data=hf_result["active_connectors"],
                                compression="gzip")
        if "per_connector_peak" in hf_result:
            grp.create_dataset("hf_per_connector_peak",
                                data=np.array(hf_result["per_connector_peak"]))


class ExperimentRunner:
    """
    Manages batch execution of paired simulations and dataset storage.

    Parameters
    ----------
    output_dir : str    Directory for CSV and HDF5 outputs
    n_workers  : int    Number of parallel processes (default: 1 for safety)
                        Set to os.cpu_count() - 1 for batch runs.
                        NOTE: PyBullet is not thread-safe; use processes.
    """

    def __init__(self, output_dir="data/results", n_workers=1):
        self.output_dir = output_dir
        self.n_workers  = n_workers
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path   = os.path.join(output_dir, "dataset.csv")
        self.hdf5_path  = os.path.join(output_dir, "timeseries.h5")

    def run_batch(self, conditions, phase_label="main",
                  save_timeseries=True):
        """
        Run paired simulations for a list of conditions.

        Parameters
        ----------
        conditions : list of dicts
            Each dict must have keys: case_id, S, mu, delta, seed
        phase_label : str
            Added to each record for train/test split tracking.
        save_timeseries : bool
            Whether to save force/position time series to HDF5.

        Returns
        -------
        pd.DataFrame   All records from this batch.
        """
        records = []
        print(f"\nRunning {len(conditions)} paired simulations "
              f"({phase_label}, {self.n_workers} worker(s))...")

        if self.n_workers == 1:
            # Serial execution — easier to debug
            for cond in tqdm(conditions, desc=phase_label):
                record, lf_res, hf_res = run_paired(**cond)
                record["phase"] = phase_label
                records.append(record)
                if save_timeseries:
                    save_time_series(self.hdf5_path, cond["case_id"],
                                     lf_res, hf_res)
        else:
            # Parallel execution using separate processes
            # Each process creates its own PyBullet client (DIRECT mode)
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(run_paired, **cond): cond
                    for cond in conditions
                }
                for future in tqdm(as_completed(futures),
                                   total=len(futures), desc=phase_label):
                    cond = futures[future]
                    try:
                        record, lf_res, hf_res = future.result()
                        record["phase"] = phase_label
                        records.append(record)
                        if save_timeseries:
                            save_time_series(self.hdf5_path,
                                             cond["case_id"], lf_res, hf_res)
                    except Exception as e:
                        print(f"Case {cond['case_id']} failed: {e}")

        df = pd.DataFrame(records)

        # Append to CSV (create if new, append if exists)
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=write_header, index=False)

        print(f"  Completed {len(records)}/{len(conditions)} cases.")
        print(f"  Adequate rate: "
              f"{df['adequate_label'].mean()*100:.1f}%")
        print(f"  Success rate (HF): "
              f"{df['hf_success'].mean()*100:.1f}%")
        print(f"  Mean speedup (HF/LF): "
              f"{df['speedup_ratio'].mean():.2f}x")

        return df

    def load_dataset(self):
        """Load the full accumulated dataset from CSV."""
        return pd.read_csv(self.csv_path)

    def load_timeseries(self, case_id):
        """Load time series for a specific case from HDF5."""
        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[f"case_{case_id:05d}"]
            return {k: grp[k][:] for k in grp.keys()}