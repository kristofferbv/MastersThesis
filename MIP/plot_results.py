import os
import re
import matplotlib.pyplot as plt

# Directory containing the files
directory = os.path.join(os.path.dirname(__file__), 'results_exponential_beta')

# Regular expression pattern to extract beta-values from file names
pattern = r'beta_(\d+\.\dt+)'

beta_values = []
last_mean_costs = []

# Iterate over files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    # Extract the beta-value from the file name
    match = re.search(pattern, filename)
    if match:
        beta_value = float(match.group(1))
        beta_values.append(beta_value)
    
    # Read the file and extract the last mean cost value
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # Find the line with the last mean cost value
        last_mean_cost_line = None
        for line in lines:
            if not line.startswith('The standard deviations') and not line.startswith('The average forecasting errors') and not line.startswith('The average optimality gap'):
                last_mean_cost_line = line
        
        # Extract the last mean cost value
        if last_mean_cost_line:
            last_mean_cost = float(last_mean_cost_line.strip())
            last_mean_costs.append(last_mean_cost)

# Plotting
plt.plot(beta_values, last_mean_costs, 'bo')
plt.xlabel('Beta Values')
plt.ylabel('Last Mean Cost')
plt.title('Last Mean Cost vs. Beta Values')
plt.show()
