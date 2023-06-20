import os
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

product_combination = "beta0-0-4-0"  # modify as needed
methods = ["exp", "con", "lin"]
seeds = ["seed0", "seed2"]
results = {}

for seed in seeds:
    results[seed] = {}
    for method in methods:
        folder = f"{method}-{product_combination}-{seed}"
        
        folder_path = os.path.join(os.path.dirname(__file__), folder)

        beta_total_costs = {}

        for filename in os.listdir(folder_path):
            if filename.startswith('simulation_output'):
                match = re.search(r'beta(\d+(\.\d+)?)', filename)
                if match:
                    beta = float(match.group(1))

                    with open(os.path.join(folder_path, filename), 'r') as file:
                        file_contents = file.read()
                        match_costs = re.search(r'Total costs for each period are: \[(.*?)\]', file_contents)
                        if match_costs:
                            costs = match_costs.group(1).split(', ')
                            costs = [float(cost) for cost in costs]
                            if beta in beta_total_costs:
                                beta_total_costs[beta].extend(costs)
                            else:
                                beta_total_costs[beta] = costs

        # Find beta with lowest average total cost and calculate that average cost
        lowest_beta = min(beta_total_costs, key=lambda beta: statistics.mean(beta_total_costs[beta]))
        average_cost = statistics.mean(beta_total_costs[lowest_beta])
        results[seed][method] = (lowest_beta, average_cost)

# Plot results
plt.figure(figsize=(10, 6))

# Sort the keys (seeds) for plotting
for i, seed in enumerate(sorted(results.keys())):
    values = [results[seed][method][1] for method in methods]  # use the average cost for plotting
    plt.plot(methods, values, marker='o', label=seed)

    # Add text note for each point
    for j, method in enumerate(methods):
        plt.text(method, values[j], f"Beta: {results[seed][method][0]}", fontsize=8, verticalalignment='bottom')

plt.title(f'Average total cost with lowest beta for product combination {product_combination}')
plt.xlabel('Method')
plt.ylabel('Average Total Cost')
plt.grid()
plt.legend(title='Seeds')
plt.show()

'''
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

product_combinations = ["beta0-0-0-4", "beta0-0-4-0", "beta0-4-0-0", "beta4-0-0-0"]
methods = ["exp", "con", "lin"]
results = {}

for combo in product_combinations:
    results[combo] = {}
    for method in methods:
        folder = f"{method}-{combo}-seed2"
        
        folder_path = os.path.join(os.path.dirname(__file__), folder)

        beta_total_costs = {}

        for filename in os.listdir(folder_path):
            if filename.startswith('simulation_output'):
                match = re.search(r'beta(\d+(\.\d+)?)', filename)
                if match:
                    beta = float(match.group(1))

                    with open(os.path.join(folder_path, filename), 'r') as file:
                        file_contents = file.read()
                        match_costs = re.search(r'Total costs for each period are: \[(.*?)\]', file_contents)
                        if match_costs:
                            costs = match_costs.group(1).split(', ')
                            costs = [float(cost) for cost in costs]
                            if beta in beta_total_costs:
                                beta_total_costs[beta].extend(costs)
                            else:
                                beta_total_costs[beta] = costs

        # Find beta with lowest average total cost and calculate that average cost
        lowest_beta = min(beta_total_costs, key=lambda beta: statistics.mean(beta_total_costs[beta]))
        average_cost = statistics.mean(beta_total_costs[lowest_beta])
        results[combo][method] = (lowest_beta, average_cost)

# Plot results
plt.figure(figsize=(10, 6))

# Sort the keys (product combinations) for plotting
for i, combo in enumerate(sorted(results.keys())):
    values = [results[combo][method][1] for method in methods]  # use the average cost for plotting
    plt.plot(methods, values, marker='o', label=combo)

    # Add text note for each point
    for j, method in enumerate(methods):
        plt.text(method, values[j], f"Beta: {results[combo][method][0]}", fontsize=8, verticalalignment='bottom')

plt.title('Average total cost with lowest beta for each product combination')
plt.xlabel('Method')
plt.ylabel('Average Total Cost')
plt.grid()
plt.legend(title='Product combinations')
plt.show()

'''
