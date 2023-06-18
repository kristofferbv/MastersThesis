import re
import numpy as np
from scipy import stats
import ast


# Read the file
with open('simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta1_seed0.txt', 'r') as f:
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

print(get_mean_service_levels('simulation_output_p2_er2_sm0_in0_lu0_t13_ep100_S2500_r1.2_beta1_seed0.txt'))