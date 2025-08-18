import pandas as pd
import numpy as np

file_path = "/home/atsushi/workspace/rigid-body-manipulation/debug_log/dev-jointpos_limits/data/data_20250817_192742.csv"
df = pd.read_csv(file_path)

print(f"Checking for NaN/Inf in {file_path}")

error_prefixes = ["qpos_error", "qvel_error", "qacc_error"]

for prefix in error_prefixes:
    print(f"\n--- Checking {prefix} columns ---")
    for i in range(6):
        col_name = f"{prefix}_{i}"
        if col_name in df.columns:
            nan_count = df[col_name].isnull().sum()
            inf_count = np.isinf(df[col_name]).sum()
            if nan_count > 0:
                print(f"Column {col_name} contains {nan_count} NaN values.")
            if inf_count > 0:
                print(f"Column {col_name} contains {inf_count} Inf values.")
            if nan_count == 0 and inf_count == 0:
                print(f"Column {col_name} contains no NaN or Inf values.")
        else:
            print(f"Column {col_name} not found in CSV.")