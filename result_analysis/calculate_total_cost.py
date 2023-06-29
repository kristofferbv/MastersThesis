import re
import numpy as np
from scipy import stats
import ast
#stochasstic vs MIP comparison:
#det
# file_path = 'det_vs_stoch/simulation_output_p4_er1_sm1_in1_lu1_t13_ep100_S2500_r1.2_beta1_seed2.txt'
#stoch
# file_path = 'det_vs_stoch/costs_simulation_output_p4_er1_sm1_in1_lu1_t13_ep100_S2500_r1.2_beta1_seed2.txt'

# file_path = 'det_vs_stoch/simulation_output_p16_er4_sm4_in4_lu4_t13_ep100_S2500_r1.2_beta1_seed2.txt'
# file_path = 'det_vs_stoch/simulation_output_p16_er4_sm4_in4_lu4_t13_ep100_S2500_r1.2_beta0.995_seed2.txt'

# file_path = 'simulation_output_p16_er4_sm4_in4_lu4_t13_ep100_S2500_r1.2_beta1_seed2.txt'
# file_path = '4_4_4_4_deterministic_seed_2.txt'
# file_path = '1_1_1_1_deterministic_seed_0.txt'



# Different costs:
#erratic major
file_path = 'compare_with_rl_erratic/compare_with_RL_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta0.05_seed0.txt'
# file_path = 'costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S1250_r1.2_beta1_seed0.txt'
# #erratic minor:
# file_path = 'compare_with_rl_erratic/compare_with_RL_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r0.6_beta0.05_seed0.txt'
# file_path = 'compare_with_rl_erratic/compare_with_RL_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r2.4_beta0.05_seed0.txt'


# The MIP vs RL comparison:
# file_path = 'costs_simulation_output_p2_er0_sm0_in2_lu0_t52_ep100_S2500_r1.2_beta1_seed0.txt'
# file_path = 'costs_simulation_output_p2_er0_sm0_in0_lu2_t13_ep100_S2500_r1.2_beta1_seed2.txt'
# file_path = 'costs_simulation_output_p2_er0_sm2_in0_lu0_t13_ep100_S2500_r1.2_beta1_seed0.txt'
# file_path = 'costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta1_seed0.txt'
# file_path = 'costs_simulation_output_p2_er0_sm0_in0_lu2_t13_ep100_S2500_r1.2_beta1_seed2.txt'


# file_path = 'costs_simulation_output_p2_er0_sm0_in2_lu0_t52_ep100_S2500_r1.2_beta1_seed0.txt'
# file_path = 'simulation_output_p2_er0_sm2_in0_lu0_t13_ep100_S2500_r1.2_beta0.085_seed1.txt'
# file_path = 'simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta0.995_seed0.txt'


#4 products comparison:
file_path = 'costs-exp-beta-4-0-0-0/costs_simulation_output_p4_er4_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta0.96_seed2.txt'



#ordering frequency comparison:
# file_path = 'varying costs -erratic 2 - without warm-up/costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep2_S1250_r1.2_beta1_seed2.txt'
# file_path = 'varying costs -erratic 2 - without warm-up/costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep2_S2500_r0.6_beta1_seed2.txt'
# file_path = 'varying costs -erratic 2 - without warm-up/costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep2_S2500_r1.2_beta1_seed2.txt'
# file_path = 'varying costs -erratic 2 - without warm-up/costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep2_S1250_r1.2_beta1_seed2.txt'
# file_path = 'varying costs -erratic 2 - without warm-up/costs_simulation_output_p2_er2_sm0_in0_lu0_t13_ep2_S5000_r1.2_beta1_seed2.txt'


with open(file_path, 'r') as f:
    content = f.read()
match = re.search(r'Total costs for each period are: \[(.*?)\]', content)

if match:
    costs_str = match.group(1)

    # Convert the list of strings to list of floats
    costs = list(map(float, costs_str.split(',')))

    # Calculate the mean
    mean = np.mean(costs)

    # Calculate the standard error
    se = stats.sem(costs)

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
joint_counts = []

# Process each line
orders = {}
orders[0] = []
orders[1] = []
count = 0
for line in lines:
    # Use regular expressions to find the episode number and the dictionary of actions
    match = re.match(r"Actions for episode (\d+) are: (.*)", line)
    joint_count = 0
    if match:
        count += 1
        episode = int(match.group(1))
        actions_str = match.group(2)

        # Convert the string of actions into a Python dictionary
        actions = ast.literal_eval(actions_str)

        # Iterate through each action, summing up the values
        for action, values in actions.items():
            has_joint = True
            for key, value in values.items():
                print(key)
                if value == 0:
                    orders[key].append(0)
                    episode_zero_counts[episode][key] += 1
                    has_joint = False
                else:
                    orders[key].append(1)
                    episode_sums[episode][key] += value
                    episode_counts[episode][key] += 1
            if has_joint:
                joint_count += 1
    joint_counts.append(joint_count)
    if count == 1:
        break


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
    print(f"For action {key}, average order quantity over all episodes: {52-overall_avg}, number of order frequency: {52-avg_zero_counts}")
print(f"joint ordering count is {sum(joint_counts)/100}")


def get_mean_from_text(pattern, text):
    match = re.search(pattern, text)
    if match:
        str_list = match.group(1)
        list_values = list(map(float, str_list.split(',')))
        return np.mean(list_values)
    return None

# Read your .txt file
with open(file_path, 'r') as f:
    content = f.read()
print("ORDERS:")
print(orders)

# Define patterns
holding_costs_pattern = r'Holding costs for each period are: \[(.*?)\]'
shortage_costs_pattern = r'Shortage costs for each period are: \[(.*?)\]'
setup_costs_pattern = r'Setup costs for each period are: \[(.*?)\]'
total_costs_pattern = r'Total costs for each period are: \[(.*?)\]'


# Get means
mean_holding_costs = get_mean_from_text(holding_costs_pattern, content)
mean_shortage_costs = get_mean_from_text(shortage_costs_pattern, content)
mean_setup_costs = get_mean_from_text(setup_costs_pattern, content)
total_costs = get_mean_from_text(total_costs_pattern, content)


# Print means
print(f'The mean holding costs are {mean_holding_costs}')
print(f'The mean shortage costs are {mean_shortage_costs}')
print(f'The mean setup costs are {mean_setup_costs}')
print(f'The mean total costs are {total_costs}')



