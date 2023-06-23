import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'


def compute_average(lst):
    return sum(lst) / len(lst) if lst else None

folders = ["costs-exp-beta-4-0-0-0"]

for folder in folders:
    numbers_part = folder.split('-beta-')[1]
    erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('-'))

    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
    
    beta_costs = []
    beta_std_errors = []

    for filename in os.listdir(folder_path):
        if filename.startswith('costs_simulation_output'):
            match = re.search(r'_beta(\d+(\.\d+)?)_', filename)
            if match:
                beta = float(match.group(1))

                with open(os.path.join(folder_path, filename), 'r') as file:
                    file_contents = file.read()
                    match_costs = re.search(r'Total costs for each period are: \[(.*?)\]', file_contents)
                    if match_costs:
                        costs = [float(cost) for cost in match_costs.group(1).split(', ')]
                        avg_cost = np.mean(costs)
                        std_error = stats.sem(costs)   # calculate the standard error
                        beta_costs.append((beta, avg_cost))
                        beta_std_errors.append((beta, std_error))

    if beta_costs:
        beta_costs.sort(key=lambda x: x[0])
        beta_std_errors.sort(key=lambda x: x[0])

        betas, costs = zip(*beta_costs)
        
        betas, std_errors = zip(*beta_std_errors)

        # calculate the 95% confidence interval
        ci = [1.96 * se for se in std_errors]

        plt.figure(figsize=(10, 6))
        plt.errorbar(betas, costs, yerr=ci, fmt='o', capsize=5)  # use errorbar function
        plt.title(f'Average Total Costs for Different Beta Values\nErratic: {erratic}, Smooth: {smooth}, Intermittent: {intermittent}, Lumpy: {lumpy}')
        plt.xlabel('Beta Value')
        plt.ylabel('Average Total Costs')
        plt.grid()
        plt.show()

    else:
        print(f"No beta and cost data in folder: {folder}")
