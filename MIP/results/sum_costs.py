import os
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'


def compute_average(lst):
    return sum(lst) / len(lst) if lst else None

folders = ["costs_exp-beta4-0-0-0"]

for folder in folders:
    # extract the product types from the folder name
    numbers_part = folder.split('-beta')[1]
    erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('-'))

    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
    
    beta_costs = []

    for filename in os.listdir(folder_path):
        if filename.startswith('costs_simulation_output'):
            # parse the beta value from the filename
            match = re.search(r'_beta(\d+(\.\d+)?)_', filename)
            if match:
                beta = float(match.group(1))

                with open(os.path.join(folder_path, filename), 'r') as file:
                    file_contents = file.read()
                    # find the list of holding costs in the file contents
                    match_costs = re.search(r'Shortage costs for each period are: \[(.*?)\]', file_contents)
                    if match_costs:
                        costs = [float(cost) for cost in match_costs.group(1).split(', ')]
                        avg_cost = np.mean(costs)  # calculate the average cost
                        beta_costs.append((beta, avg_cost))

    if beta_costs:
        # sort by beta value
        beta_costs.sort(key=lambda x: x[0])

        # unzip the list of tuples into two lists
        betas, costs = zip(*beta_costs)

        # create a line plot for costs
        plt.figure(figsize=(10, 6))
        plt.plot(betas, costs, marker='o')   # use 'o' to mark the data points
        plt.title(f'Average Shortage Costs for Different Beta Values\nErratic: {erratic}, Smooth: {smooth}, Intermittent: {intermittent}, Lumpy: {lumpy}')
        plt.xlabel('Beta Value')
        plt.ylabel('Average Shortage Cost')
        plt.grid()
        plt.show()

    else:
        print(f"No beta and cost data in folder: {folder}")
