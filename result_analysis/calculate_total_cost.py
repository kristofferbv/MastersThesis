import re
import numpy as np
from scipy import stats
import ast
#stochasstic vs MIP comparison:
# file_path = 'simulation_output_p4_er1_sm1_in1_lu1_t13_ep100_S2500_r1.2_beta1_seed0.txt'

# file_path = '4_4_4_4_deterministic_seed_2.txt'
file_path = '1_1_1_1_deterministic_seed_0.txt'



# The MIP vs RL comparison:
# file_path = 'simulation_output_p2_er0_sm0_in2_lu0_t13_ep10_S2500_r1.2_beta1_seed0.txt'
# file_path = 'simulation_output_p2_er0_sm2_in0_lu0_t13_ep100_S2500_r1.2_beta0.085_seed1.txt'
# file_path = 'simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta0.05_seed0.txt'
with open(file_path, 'r') as f:
    content = f.read()
match = re.search(r'Total costs for each period are: \[(.*?)\]', content)

if match:
    costs_str = match.group(1)

    # Convert the list of strings to list of floats
    costs = list(map(float, costs_str.split(',')))

    # Calculate the mean
    mean = np.mean(costs[1:])

    # Calculate the standard error
    se = stats.sem(costs[1:])

    # Calculate the confidence interval
    confidence = 0.95
    ci = se * stats.t.ppf((1 + confidence) / 2., len(costs)-1)

    print(f'The mean is {mean} with a 95% confidence interval of +/- {ci}')
else:
    print("No match found")

def get_mean_service_levels(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # find the start and end of dictionary
    start = content.index("Service levels are: ") + len("Service levels are: ")
    end = content.index('\n', start)
    dict_string = content[start:end]

    # convert string to dictionary
    service_levels = ast.literal_eval(dict_string)

    mean_service_levels = {}

    for product_index, levels in service_levels.items():
        mean_service_levels[product_index] = np.mean(levels)

    return mean_service_levels

print(get_mean_service_levels(file_path))

import re
import ast
import collections

# Open your text file
with open(file_path, "r") as file:
    lines = file.readlines()

# Initialize dictionaries to hold the sums, counts and zero_counts for each episode
episode_sums = collections.defaultdict(lambda: collections.defaultdict(int))
episode_counts = collections.defaultdict(lambda: collections.defaultdict(int))
episode_zero_counts = collections.defaultdict(lambda: collections.defaultdict(int))

# Process each line
for line in lines:
    # Use regular expressions to find the episode number and the dictionary of actions
    match = re.match(r"Actions for episode (\d+) are: (.*)", line)
    if match:
        episode = int(match.group(1))
        actions_str = match.group(2)

        # Convert the string of actions into a Python dictionary
        actions = ast.literal_eval(actions_str)

        # Iterate through each action, summing up the values
        for action, values in actions.items():
            for key, value in values.items():
                if value == 0:
                    episode_zero_counts[episode][key] += 1
                else:
                    episode_sums[episode][key] += value
                    episode_counts[episode][key] += 1

# Now calculate averages per episode and then overall average, and print the results
total_avg_sums = collections.defaultdict(int)
total_avg_counts = collections.defaultdict(int)
total_zero_counts = collections.defaultdict(int)

for episode, sums in episode_sums.items():
    for key, sum_value in sums.items():
        avg = sum_value / episode_counts[episode][key]
        total_avg_sums[key] += avg
        total_avg_counts[key] += 1
        total_zero_counts[key] += episode_zero_counts[episode][key]

for key, sum_value in total_avg_sums.items():
    overall_avg = sum_value / total_avg_counts[key]
    avg_zero_counts = total_zero_counts[key] / len(episode_sums.items())
    print(f"For action {key}, average order quantity over all episodes: {overall_avg}, number of non-orders: {avg_zero_counts}")
