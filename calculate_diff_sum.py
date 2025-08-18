import pandas as pd

file_path = "/home/atsushi/workspace/rigid-body-manipulation/debug_log/dev-jointpos_limits/data/data_20250817_182915.csv"
df = pd.read_csv(file_path)

total_abs_diff_sum_q0 = 0.0

target_col = f"tgt_qpos_0"
actual_col = f"act_qpos_0"
if target_col in df.columns and actual_col in df.columns:
    total_abs_diff_sum_q0 = (df[target_col] - df[actual_col]).abs().sum()
else:
    print(f"Warning: Columns {target_col} or {actual_col} not found in the CSV.")

print(f"{total_abs_diff_sum_q0:.15f}")