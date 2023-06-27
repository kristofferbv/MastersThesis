import os
import re
import matplotlib.pyplot as plt

from matplotlib import font_manager
font_manager._rebuild()

plt.rcParams['font.family'] = 'CMU Concrete'

files = ["exp-beta-simulation_output_p4_er4_sm0_in0_lu0_t13_ep1000_S2500_r1.2_beta0.995_seed2.txt",
         "con-beta-simulation_output_p4_er4_sm0_in0_lu0_t13_ep1000_S2500_r1.2_beta0.965_seed2.txt",
         "linear-beta-simulation_output_p4_er4_sm0_in0_lu0_t13_ep1000_S2500_r1.2_beta0.0_seed2.txt"]  # replace with your file names

data = []

current_dir = os.path.dirname(os.path.abspath(__file__))


for filename in files:
    beta_total_costs = {}

    # parse the beta value from the filename
    match = re.search(r'beta(\d+(\.\d+)?)', filename)
    if match:
        beta = float(match.group(1))

        file_path = os.path.join(current_dir, filename)

        with open(file_path, 'r') as file:
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
    
    # appending all costs to a single data array
    data.append([cost for sublist in beta_total_costs.values() for cost in sublist])

# create box plots for each beta value
plt.figure(figsize=(10, 6))

custom_blue = (0.25, 0.5, 1)  # RGB tuple representing a color in between 'blue' and 'lightblue'
bp = plt.boxplot(data, labels=['exponential: 0.995' , 'constant: 0.965', 'linear: 0.0'], showfliers=False, patch_artist=True, medianprops={'color': 'black'},
                 boxprops={'facecolor': custom_blue, 'edgecolor': custom_blue},
                 whiskerprops={'color': custom_blue}, capprops={'color': custom_blue},
                 flierprops={'markeredgecolor': custom_blue})


plt.title(f'Boxplot of Total Costs for the Best Values of each Parametrization')
plt.xlabel('Parametrization method: value')
plt.ylabel('Total Costs')
plt.grid()
plt.show()
