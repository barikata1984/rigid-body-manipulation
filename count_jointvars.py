import json

file_path = "configurations/trajectories/optimal-excitation.json"

try:
    with open(file_path) as f:
        data = json.load(f)

    if "jointvars" in data and isinstance(data["jointvars"], list):
        count = len(data["jointvars"])
        print(f"The 'jointvars' array has {count} elements.")
    else:
        print("'jointvars' key not found or is not a list.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {file_path}. Please ensure it's a valid JSON file.")


import pdb

pdb.set_trace()
