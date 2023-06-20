import os
import re
import matplotlib.pyplot as plt
import numpy as np

import os
import re
import matplotlib.pyplot as plt
import numpy as np

folders = ["exp-beta0-0-0-4-seed2", "exp-beta0-0-0-4-seed0" , "exp-beta0-0-4-0-seed2", "exp-beta0-0-4-0-seed0" , "exp-beta0-4-0-0-seed2", "exp-beta0-4-0-0-seed0", "exp-beta4-0-0-0-seed2", "exp-beta4-0-0-0-seed0", "exp-beta2-2-2-2-seed2", "exp-beta2-2-2-2-seed0"]   # the list of folders

current_dir = os.path.dirname(os.path.abspath(__file__))



for folder in folders:
    # extract the product types from the folder name
    numbers_part, seed_part = folder.split('-seed')
    erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('exp-beta')[1].split('-'))

    folder_path = os.path.join(os.path.dirname(__file__), folder)
    
    beta_costs_times = []

    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_output'):
            # parse the beta value from the filename
            match = re.search(r'beta(\d+(\.\d+)?)', filename)
            if match:
                beta = float(match.group(1))

                with open(os.path.join(folder_path, filename), 'r') as file:
                    file_contents = file.read()
                    # find the list of mean costs and average run times in the file contents
                    match_costs = re.search(r'List of mean costs for each period: \[(.*?)\]', file_contents)
                    match_time = re.search(r'The average time to run the model in Gurobi is: (\d\.\d+)', file_contents)
                    if match_costs and match_time:
                        costs = match_costs.group(1).split(', ')
                        last_cost = float(costs[-1])  # take the last cost
                        avg_time = float(match_time.group(1))  # get the average run time
                        beta_costs_times.append((beta, last_cost, avg_time))

    # sort by beta value
    beta_costs_times.sort(key=lambda x: x[0])

    # unzip the list of tuples into three lists
    betas, costs, times = zip(*beta_costs_times)

    # normalize times to range [0, 1] for color mapping
    times_norm = [(t - min(times)) / (max(times) - min(times)) for t in times]

    # create a scatter plot for costs
    plt.figure(figsize=(10, 6))
    plt.scatter(betas, costs, c=times_norm, cmap='RdYlGn_r')
    plt.title(f'Mean Costs for Different Beta Values\nErratic: {erratic}, Smooth: {smooth}, Intermittent: {intermittent}, Lumpy: {lumpy}')
    plt.xlabel('Beta Value')
    plt.ylabel('Last Mean Cost')
    cbar = plt.colorbar()
    cbar.set_label('Normalized Average Run Time')
    plt.grid()
    plt.show()

    # create a plot for average run times
    plt.figure(figsize=(10, 6))
    plt.plot(betas, times, marker='o')
    plt.title(f'Average Run Time for Different Beta Values\nErratic: {erratic}, Smooth: {smooth}, Intermittent: {intermittent}, Lumpy: {lumpy}')
    plt.xlabel('Beta Value')
    plt.ylabel('Average Run Time')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # use scientific notation for y-axis
    plt.grid()
    plt.show()
