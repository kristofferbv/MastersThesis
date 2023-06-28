import os
import re
import numpy as np

def compute_average(lst):
    return sum(lst) / len(lst) if lst else None

file_list = [
    "costs_simulation_output_sl_0.9_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta1_seed0.txt",

    # Add more file names as needed
]

script_dir = os.path.dirname(os.path.abspath(__file__))
beta_service_levels = []

for filename in file_list:
    file_path = os.path.join(script_dir, filename)

    # parse the beta value from the filename
    match = re.search(r'_beta(\d+(\.\d+)?)_', filename)
    if match:
        beta = float(match.group(1))

        with open(file_path, 'r') as file:
            file_contents = file.read()

            # find the list of service levels in the file contents
            match_service_levels = re.search(r'Service levels are: (.*?)\n', file_contents)
            if match_service_levels:
                service_levels = [float(sl) for sl in match_service_levels.group(1).split(', ')]
                avg_service_level = compute_average(service_levels)  # calculate the average service level
                beta_service_levels.append((beta, avg_service_level))

if beta_service_levels:
    # sort by beta value
    beta_service_levels.sort(key=lambda x: x[0])

    print("Average Service Levels:")
    for beta, avg_service_level in beta_service_levels:
        print(f"Beta: {beta}, Average Service Level: {avg_service_level}")

else:
    print("No service level data found.")
