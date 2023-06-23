import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


import os
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'


folders = ["lin-beta0-0-0-4-seed2",  "lin-beta0-0-4-0-seed2", "lin-beta0-4-0-0-seed2",  "lin-beta4-0-0-0-seed2"]   # the list of folders

current_dir = os.path.dirname(os.path.abspath(__file__))




for folder in folders:
    # extract the product types from the folder name
    numbers_part, seed_part = folder.split('-seed')
    erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('-beta')[1].split('-'))

    folder_path = os.path.join(os.path.dirname(__file__), folder)
    
    beta_costs_conf_intervals = []

    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_output'):
            # parse the beta value from the filename
            match = re.search(r'beta(\d+(\.\d+)?)', filename)
            if match:
                beta = float(match.group(1))

                with open(os.path.join(folder_path, filename), 'r') as file:
                    file_contents = file.read()
                    # find the list of total costs and mean costs in the file contents
                    match_total_costs = re.search(r'Total costs for each period are: \[(.*?)\]', file_contents)
                    match_costs = re.search(r'List of mean costs for each period: \[(.*?)\]', file_contents)
                    if match_total_costs and match_costs:
                        total_costs = list(map(float, match_total_costs.group(1).split(', ')))
                        costs = match_costs.group(1).split(', ')
                        last_cost = float(costs[-1])  # take the last cost
                        
                        mean_total_cost = np.mean(total_costs)
                        total_cost_conf_interval = stats.t.interval(alpha=0.95, df=len(total_costs)-1, loc=mean_total_cost, scale=stats.sem(total_costs))  # calculate the 95% confidence interval

                        beta_costs_conf_intervals.append((beta, last_cost, total_cost_conf_interval))

    # sort by beta value
    beta_costs_conf_intervals.sort(key=lambda x: x[0])

    # unzip the list of tuples into three lists
    betas, costs, conf_intervals = zip(*beta_costs_conf_intervals)
    cost_lower, cost_upper = zip(*conf_intervals)

    # create a scatter plot for costs with 95% confidence interval error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(betas, costs, yerr=[cost_upper-np.array(costs), np.array(costs)-cost_lower], fmt='o')  # add error bars for the confidence intervals
    plt.title(f'Mean Total Costs for Different Beta Values\nErratic: {erratic}, Smooth: {smooth}, Intermittent: {intermittent}, Lumpy: {lumpy}')
    plt.xlabel('Beta Value')
    plt.ylabel('Mean Total Costs')
    plt.grid()
    plt.show()
