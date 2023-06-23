import os
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'


# the list of folders, split into different categories
exp_folders = ["exp-beta0-0-0-4-seed2", "exp-beta0-0-0-4-seed0", "exp-beta0-0-4-0-seed2", "exp-beta0-0-4-0-seed0", "exp-beta0-4-0-0-seed2", "exp-beta0-4-0-0-seed0", "exp-beta4-0-0-0-seed2", "exp-beta4-0-0-0-seed0"]
con_folders = ["con-beta0-0-0-4-seed2", "con-beta0-0-0-4-seed0", "con-beta0-0-4-0-seed2", "con-beta0-0-4-0-seed0", "con-beta0-4-0-0-seed2", "con-beta0-4-0-0-seed0", "con-beta4-0-0-0-seed2", "con-beta4-0-0-0-seed0"]
lin_folders = ["lin-beta0-0-0-4-seed2", "lin-beta0-0-0-4-seed0", "lin-beta0-0-4-0-seed2", "lin-beta0-0-4-0-seed0", "lin-beta0-4-0-0-seed2", "lin-beta0-4-0-0-seed0", "lin-beta4-0-0-0-seed2", "lin-beta4-0-0-0-seed0"]

# dictionaries mapping folder lists to colors
folder_dict = {0: exp_folders, 1: con_folders, 2: lin_folders}
color_dict = {0: 'b', 1: 'r', 2: 'g'}

current_dir = os.path.dirname(os.path.abspath(__file__))

# create the plots
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Mean Costs for Different Beta Values')
ax1.set_xlabel('Beta Value')
ax1.set_ylabel('Last Mean Cost')
ax1.grid()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title('Average Run Time for Different Beta Values')
ax2.set_xlabel('Beta Value')
ax2.set_ylabel('Average Run Time')
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.grid()

for color, folders in folder_dict.items():
    for folder in folders:
        # extract the product types from the folder name
        numbers_part, seed_part = folder.split('-seed')
        erratic, smooth, intermittent, lumpy = map(int, numbers_part.split('-beta')[1].split('-'))

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

        # plot the data for this folder with the appropriate color
        ax1.scatter(betas, costs, color=color_dict[color])
        ax2.plot(betas, times, color=color_dict[color], marker='o')

# show the plots
plt.show()
