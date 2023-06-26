import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='CMU Concrete')

# the rest of your plotting code


# Hypothetical DataFrames (Replace with your actual data)
df_total_costs = pd.DataFrame({
    'Setup Costs': ['Base Major', 'Half Major', 'Double Major', 'Half Minor', 'Double Minor'],
    'RL Total Costs': [210532.5, 161437.6, 283887.4, 192308.3, 259400.9],
    'MIP Total Costs': [221357.6, 163089.2, 303007.3, 200648.6, 261779.7],
    'RL Confidence Interval': [1062.0, 500.7, 952.2, 992.9, 942.9],
    'MIP Confidence Interval': [1707.9, 1873.7, 2067.2, 1662.6, 1886.6]
})


df_service_levels = pd.DataFrame({
    'Setup Costs': ['Base Major', 'Half Major', 'Double Major', 'Half Minor'],
    'RL Product 1': [98.6, 98.9, 98.9, 99.0],
    'RL Product 2': [99.0, 99.4, 98.6, 97.8],
    'MIP Product 1': [99.9, 99.5, 99.9, 99.8],
    'MIP Product 2': [99.9, 99.8, 99.9, 99.9],
})

# Bar chart
plt.figure(figsize=(10,6))
plt.bar(df_total_costs['Setup Costs'], df_total_costs['RL Total Costs'], label='RL')
plt.bar(df_total_costs['Setup Costs'], df_total_costs['MIP Total Costs'], label='MIP', alpha=0.5)
plt.legend()
plt.title('Total Costs Comparison Between RL and MIP')
plt.xlabel('Setup Costs')
plt.ylabel('Total Costs (NOK)')
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_service_levels.set_index('Setup Costs'), annot=True, cmap='YlGnBu')
plt.title('Achieved Service Levels under RL and MIP Approaches')
plt.show()


# Assuming you have your means and confidence intervals in the lists as follows
RL_means = df_total_costs['RL Total Costs']
MIP_means = df_total_costs['MIP Total Costs']

RL_confidence_intervals = df_total_costs['RL Confidence Interval']
MIP_confidence_intervals = df_total_costs['MIP Confidence Interval']

ind = np.arange(len(RL_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(ind - width/2, RL_means, width, yerr=RL_confidence_intervals,
                label='RL', color='blue', capsize=10)
rects2 = ax.bar(ind + width/2, MIP_means, width, yerr=MIP_confidence_intervals,
                label='MIP', color='cyan', capsize=10)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Total Costs (NOK)')
ax.set_title('Total Costs Comparison Between RL and MIP')
ax.set_xticks(ind)
ax.set_xticklabels(df_total_costs['Setup Costs'])
ax.legend()

fig.tight_layout()

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))

sns.set_style("whitegrid")

# Draw the bars for RL and MIP and include error bars manually
bar1 = sns.barplot(x='Setup Costs', y='RL Total Costs', data=df_total_costs, label='RL', color='b', capsize=.1)
bar2 = sns.barplot(x='Setup Costs', y='MIP Total Costs', data=df_total_costs, label='MIP', color='r', alpha=0.7, capsize=.1)

# Adding error bars manually
for i in range(df_total_costs.shape[0]):
    bar1.errorbar(i, df_total_costs['RL Total Costs'].iloc[i], yerr=df_total_costs['RL Confidence Interval'].iloc[i], fmt='-', color='black')
    bar2.errorbar(i, df_total_costs['MIP Total Costs'].iloc[i], yerr=df_total_costs['MIP Confidence Interval'].iloc[i], fmt='-', color='black')

# Add a legend and informative axis label
ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(xlim=(-0.5, 4.5), ylabel="Total Costs (NOK)", xlabel="Setup Costs")
sns.despine(left=True, bottom=True)

plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))

sns.set_style("whitegrid")

# Adjust bar width
barWidth = 0.35

# Define bar positions
r1 = np.arange(len(df_total_costs))
r2 = [x + barWidth for x in r1]

# Draw the bars for RL and MIP and include error bars manually
bar1 = plt.bar(r1, df_total_costs['RL Total Costs'], width=barWidth, label='RL', color='b', yerr=df_total_costs['RL Confidence Interval'], capsize=7)
bar2 = plt.bar(r2, df_total_costs['MIP Total Costs'], width=barWidth, label='MIP', color='r', alpha=0.7, yerr=df_total_costs['MIP Confidence Interval'], capsize=7)

# Add xticks on the middle of the grouped bars
plt.title('Total Costs Comparison Between RL and MIP')
plt.xlabel('Setup Costs', fontweight='bold')
plt.ylabel('Total Costs (NOK)')
plt.xticks([r + barWidth / 2 for r in range(len(df_total_costs))], df_total_costs['Setup Costs'])

# Create legend & Show graphic
plt.legend()
plt.show()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constructing the data
data = {
    "Product": ["Erratic", "Smooth", "Intermittent", "Lumpy"],
    "RL - Product 1": [98.6, 94.5, 91.1, 96.5],
    "RL - Product 2": [99.0, 92.3, 96.0, 94.3],
    "MIP - Product 1": [99.9, 100, 99.0, 98.6],
    "MIP - Product 2": [99.9, 96.0, 97.3, 97.9]
}

df = pd.DataFrame(data).set_index('Product')

# Convert the data to percentage format for annotations
annot = df.applymap(lambda x: f'{x:.1f}%')

# Creating the heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=annot, cmap="YlGnBu", fmt='', linewidths=.5)

plt.title('Achieved Service Level for RL and MIP Methods Across Different Products')
plt.xlabel('Method and Product')
plt.ylabel('Demand Type')
plt.show()


