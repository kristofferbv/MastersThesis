import os
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'


folder = "time_periods"  # specify the folder

folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)

time_periods_costs_times = []

for filename in os.listdir(folder_path):
    if filename.startswith('simulation_output'):
        # parse the number of time periods from the filename
        match = re.search(r'_t(\d+)_', filename)
        if match:
            time_periods = int(match.group(1))

            with open(os.path.join(folder_path, filename), 'r') as file:
                file_contents = file.read()
                # find the list of mean costs and average run times in the file contents
                match_costs = re.search(r'List of mean costs for each period: \[(.*?)\]', file_contents)
                match_time = re.search(r'The average time to run the model in Gurobi is: (\d\.\d+)', file_contents)
                if match_costs and match_time:
                    costs = match_costs.group(1).split(', ')
                    last_cost = float(costs[-1])  # take the last cost
                    avg_time = float(match_time.group(1))  # get the average run time
                    time_periods_costs_times.append((time_periods, last_cost, avg_time))

# sort by the number of time periods
time_periods_costs_times.sort(key=lambda x: x[0])

# unzip the list of tuples into three lists
time_periods, costs, times = zip(*time_periods_costs_times)

# normalize times to range [0, 1] for color mapping
times_norm = [(t - min(times)) / (max(times) - min(times)) for t in times]

# create a scatter plot for costs
plt.figure(figsize=(10, 6))
plt.scatter(time_periods, costs)
plt.title('Mean Total Costs for Different Number of Time Periods')
plt.xlabel('Number of Time Periods')
plt.ylabel('Mean Total Costs')
plt.grid()
plt.show()

# create a plot for average run times
plt.figure(figsize=(10, 6))
plt.plot(time_periods, times, marker='o')
plt.title('Average Run Time for Different Number of Time Periods')
plt.xlabel('Number of Time Periods')
plt.ylabel('Average Run Time')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # use scientific notation for y-axis
plt.grid()
plt.show()
