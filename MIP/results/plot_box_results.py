import os
import re
import matplotlib.pyplot as plt
import numpy as np

folders = ["exp-beta4-0-0-0-seed2"]  # the list of folders

current_dir = os.path.dirname(os.path.abspath(__file__))

for folder in folders:
    # extract the product types from the folder name
    numbers_part, seed_part = folder.split('-seed')
    erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('exp-beta')[1].split('-'))

    folder_path = os.path.join(os.path.dirname(__file__), folder)

    beta_total_costs = {}

    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_output'):
            # parse the beta value from the filename
            match = re.search(r'beta(\d+(\.\d+)?)', filename)
            if match:
                beta = float(match.group(1))

                with open(os.path.join(folder_path, filename), 'r') as file:
                    file_contents = file.read()
                    # find the total costs in the file contents
                    match_costs = re.search(r'Total costs for each period are: \[(.*?)\]', file_contents)
                    if match_costs:
                        costs = match_costs.group(1).split(', ')
                        costs = [float(cost) for cost in costs]
                        if beta in beta_total_costs:
                            beta_total_costs[beta].extend(costs)
                        else:
                            beta_total_costs[beta] = costs

    # create box plots for each beta value
    plt.figure(figsize=(10, 6))
    plt.boxplot(beta_total_costs.values(), labels=beta_total_costs.keys(), sym='.')
    plt.title(f'Boxplot of Total Costs for {folder}')
    plt.xlabel('Beta Value')
    plt.ylabel('Total Costs')
    plt.grid()
    plt.show()
