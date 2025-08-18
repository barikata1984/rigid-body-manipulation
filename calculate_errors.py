import pandas as pd
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    file_path: str
    """Path to the CSV data file."""

def main():
    args = tyro.cli(Args)
    file_path = args.file_path

    df = pd.read_csv(file_path)

    error_types = {"qpos_error": "Position Error", "qvel_error": "Velocity Error", "qacc_error": "Acceleration Error"}

    for error_prefix, error_name in error_types.items():
        print(f"\n--- {error_name} ---")
        max_abs_errors = {}
        rmse_errors = {}

        for i in range(6):
            col_name = f"{error_prefix}_{i}"
            if col_name in df.columns:
                diff = df[col_name]
                max_abs_errors[f"q{i}"] = diff.abs().max()
                rmse_errors[f"q{i}"] = np.sqrt(np.mean(diff**2))
            else:
                print(f"Warning: Column {col_name} not found in the CSV.")

        print("Max Absolute Errors:")
        for joint, error in max_abs_errors.items():
            print(f"  {joint}: {error:.15f}")

        print("RMSE Errors:")
        for joint, error in rmse_errors.items():
            print(f"  {joint}: {error:.15f}")
        print()

if __name__ == "__main__":
    main()
