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


#Bar chart 1
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

#Bar chart 2
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

#Heatmap
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


#Bar chart 3
import pandas as pd

df = pd.DataFrame({
    'Approach': ['RL', 'MIP', 'RL', 'MIP', 'RL', 'MIP', 'RL', 'MIP'],
    'Product Category': ['Erratic', 'Erratic', 'Smooth', 'Smooth', 'Intermittent', 'Intermittent', 'Lumpy', 'Lumpy'],
    'Total Costs (NOK)': [210532.504, 221357.578, 90070.864, 97679.267, 28289.221, 31162.351, 26829.678, 32199.262],
    'Holding Costs (NOK)': [95235.151, 122703.391, 61276.937, 38574.016, 13364.684, 14848.105, 6117.465, 13264.621],
    'Shortage Costs (NOK)': [29325.853, 1424.186, 14276.427, 2940.251, 3579.538, 534.246, 4701.334, 389.641],
    'Setup Costs (NOK)': [85971.5, 97230.0, 14517.5, 56165.0, 11345.0, 15780.0, 16010.88, 18545.0]
})

RL_error = [1062.0, 517.1, 286.2, 298.7]
MIP_error = [1707.9, 1282.8, 294.6, 1402.3]

categories = df['Product Category'].unique()
approaches = ['RL', 'MIP']

barWidth = 0.35

fig, ax = plt.subplots(figsize=(12, 8))

bar_positions = np.arange(len(categories))

for idx, approach in enumerate(approaches):
    holding = df[df['Approach'] == approach]['Holding Costs (NOK)']
    shortage = df[df['Approach'] == approach]['Shortage Costs (NOK)']
    setup = df[df['Approach'] == approach]['Setup Costs (NOK)']

    bar_positions_shifted = [x + idx * barWidth for x in bar_positions]

    color = 'C{}'.format(idx)  # Use color based on index to match RL and MIP bars

    # Add error bars to the top bars only
    if approach == 'RL':
        error = RL_error
    else:
        error = MIP_error

    ax.bar(bar_positions_shifted, holding, width=barWidth, label=f'Holding Costs ({approach})', color=color, alpha=1.0)
    ax.bar(bar_positions_shifted, shortage, bottom=holding, width=barWidth, label=f'Shortage Costs ({approach})', color=color, alpha=0.8)
    ax.bar(bar_positions_shifted, setup, bottom=[i + j for i, j in zip(holding, shortage)], width=barWidth, label=f'Setup Costs ({approach})', color=color, alpha=0.6)

    # Add error bars at the top of the top bars
    top_bar_values = [i + j + k for i, j, k in zip(holding, shortage, setup)]

    ax.errorbar(bar_positions_shifted, top_bar_values, yerr=error, fmt='none', ecolor='black', capsize=4)

ax.set_xlabel('Product Category')
ax.set_ylabel('Costs (NOK)')
# ax.set_title('Cost Components under RL and MIP Approaches')
ax.set_xticks([r + barWidth / 2 for r in range(len(categories))])
ax.set_xticklabels(categories)
ax.legend()

plt.show()





import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Approach': ['RL', 'MIP', 'RL', 'MIP', 'RL', 'MIP', 'RL', 'MIP', 'RL', 'MIP'],
    'Product Category': ['Base Major', 'Base Major', 'Half Major', 'Half Major', 'Double Major', 'Double Major', 'Half Minor', 'Half Minor', 'Double Minor', 'Double Minor'],
    'Total Costs (NOK)': [210532.5, 221357.6, 161437.6, 163089.2, 283887.4, 303007.3, 192308.3, 200648.6, 259400.9, 261779.7],
    'Holding Costs (NOK)': [95235.2, 122703.4, 93867.6, 80193.7, 117702.1, 163612.4, 91536.4, 109128.0, 103873.1, 138651.0],
    'Shortage Costs (NOK)': [29325.9, 1424.2, 23920.0, 6550.5, 22103.3, 604.9, 22854.2, 1933.1, 24785.4, 1008.7],
    'Setup Costs (NOK)': [75971.5, 97230.0, 43650.0, 76345.0, 144082.0, 138790.0, 77917.8, 89587.5, 130742.5, 122120.0]
})

RL_error = [1062.0, 500.7, 952.2, 992.9, 942.9]
MIP_error = [1707.9, 1873.7, 2067.2, 1662.6, 1886.6]

categories = df['Product Category'].unique()
approaches = ['RL', 'MIP']

barWidth = 0.35

fig, ax = plt.subplots(figsize=(12, 8))

bar_positions = np.arange(len(categories))

for idx, approach in enumerate(approaches):
    holding = df[df['Approach'] == approach]['Holding Costs (NOK)']
    shortage = df[df['Approach'] == approach]['Shortage Costs (NOK)']
    setup = df[df['Approach'] == approach]['Setup Costs (NOK)']

    bar_positions_shifted = [x + idx * barWidth for x in bar_positions]

    color = 'C{}'.format(idx)  # Use color based on index to match RL and MIP bars

    # Add error bars to the top bars only
    if approach == 'RL':
        error = RL_error
    else:
        error = MIP_error

    ax.bar(bar_positions_shifted, holding, width=barWidth, label=f'Holding Costs ({approach})', color=color, alpha=1.0)
    ax.bar(bar_positions_shifted, shortage, bottom=holding, width=barWidth, label=f'Shortage Costs ({approach})', color=color, alpha=0.8)
    ax.bar(bar_positions_shifted, setup, bottom=[i + j for i, j in zip(holding, shortage)], width=barWidth, label=f'Setup Costs ({approach})', color=color, alpha=0.6)

    # Add error bars at the top of the top bars
    top_bar_values = [i + j + k for i, j, k in zip(holding, shortage, setup)]

    ax.errorbar(bar_positions_shifted, top_bar_values, yerr=error, fmt='none', ecolor='black', capsize=5)

ax.set_xlabel('Setup Cost Instance')
ax.set_ylabel('Costs (NOK)')
# ax.set_title('Cost Components under RL and MIP Approaches')
ax.set_xticks([r + barWidth / 2 for r in range(len(categories))])
ax.set_xticklabels(categories)
ax.legend()

plt.show()





