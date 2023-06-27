import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='CMU Concrete')
plt.rcParams['font.size'] = 13

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
    'Total Costs (NOK)': [210532.5, 221357.6, 161437.6, 163089.2, 283887.4, 303007.3, 192308.3, 200648.6, 255467.2, 261779.7],
    'Holding Costs (NOK)': [95235.2, 122703.4, 93867.6, 80193.7, 117702.1, 163612.4, 91536.4, 109128.0, 118690.4, 138651.0],
    'Shortage Costs (NOK)': [29325.9, 1424.2, 23920.0, 6550.5, 22103.3, 604.9, 22854.2, 1933.1, 13482.7, 1008.7],
    'Setup Costs (NOK)': [75971.5, 97230.0, 43650.0, 76345.0, 144082.0, 138790.0, 77917.8, 89587.5, 123294.0, 122120.0]
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


import matplotlib.pyplot as plt

# setup costs structures
costs = ['Major double', 'Major half', 'Ratio half', 'Double minor']

# average order quantities
avg_order_quantities_0 = [96.96, 83.15, 77.50, 81.92]
avg_order_quantities_1 = [49.32, 29.24, 14.85, 41.91]

# average non-ordering periods
avg_non_order_periods_0 = [34.83, 32.2, 30.80, 31.76]
avg_non_order_periods_1 = [46.17, 42.93, 35.41, 45.17]

# calculate order frequencies
order_freq_0 = [52 - non_order for non_order in avg_non_order_periods_0]
order_freq_1 = [52 - non_order for non_order in avg_non_order_periods_1]

# plot average order quantities
plt.figure(figsize=(10, 5))
plt.bar(costs, avg_order_quantities_0, width=0.4, align='edge', label='Product 0')
plt.bar(costs, avg_order_quantities_1, width=-0.4, align='edge', label='Product 1')
plt.xlabel('Setup Costs Structure')
plt.ylabel('Average Order Quantity')
plt.title('Average Order Quantity vs Setup Costs Structure')
plt.legend()
plt.show()

# setup costs structures
costs = ['Base Case', 'Major double', 'Major half', 'Minor ratio double', 'Minor ratio half']

# average order quantities for RL model
avg_order_quantities_rl_0 = [83.722, 96.96, 83.15, 81.92, 77.50]
avg_order_quantities_rl_1 = [30.893, 49.32, 29.24, 41.91, 14.85]

# average order quantities for MIP model
avg_order_quantities_mip_0 = [74.818, 110.06, 49.17, 87.34, 63.58]
avg_order_quantities_mip_1 = [31.880, 33.62, 25.15, 35.04, 23.55]

# calculate order frequencies for RL model
avg_non_order_periods_rl_0 = [52 - 19.6, 34.83, 32.2, 31.76, 30.80]
avg_non_order_periods_rl_1 = [52 - 8.8, 46.17, 42.93, 45.17, 35.41]
order_freq_rl_0 = [52 - non_order for non_order in avg_non_order_periods_rl_0]
order_freq_rl_1 = [52 - non_order for non_order in avg_non_order_periods_rl_1]

# calculate order frequencies for MIP model
avg_non_order_periods_mip_0 = [52 - 21.8, 37.25, 19.19, 33.43, 26.47]
avg_non_order_periods_mip_1 = [52 - 7.6, 44.69, 42.33, 45.03, 41.72]
order_freq_mip_0 = [52 - non_order for non_order in avg_non_order_periods_mip_0]
order_freq_mip_1 = [52 - non_order for non_order in avg_non_order_periods_mip_1]

# average total costs
# add a placeholder for the base case cost - you will need to provide this
avg_total_costs_rl = [210532.5, 283887.36, 161437.59, 255467.16, 196989.85]
avg_total_costs_mip = [221357.6, 303007.31, 162760.16, 261779.73, 200648.57]

# Joint order frequencies for both RL and MIP models.
# Add placeholder for base case joint order frequency - you will need to provide this.
joint_order_freq_rl = [min(order_freq_rl_0[0], order_freq_rl_1[0]), min(order_freq_rl_0[1], order_freq_rl_1[1]), min(order_freq_rl_0[2], order_freq_rl_1[2]), min(order_freq_rl_0[3], order_freq_rl_1[3]),15.369]
joint_order_freq_mip = [7.38, 7.08, 9.15, 6.46, 10.03]




import numpy as np
import matplotlib.pyplot as plt

# setup costs structures
costs = ['Base Case', 'Major double', 'Major half', 'Minor Double', 'Minor Half']
x = np.arange(len(costs))  # the label locations

width = 0.15  # the width of the bars

fig, ax = plt.subplots()

# Create some additional space between RL and MIP
space = 0.1

rects1 = ax.bar(x - width - space, avg_order_quantities_rl_0, width, label='RL Product 0', color='b')
rects2 = ax.bar(x - space, avg_order_quantities_rl_1, width, label='RL Product 1', color='b', alpha=0.5)
rects3 = ax.bar(x + space, avg_order_quantities_mip_0, width, label='MIP Product 0', color='g')
rects4 = ax.bar(x + width + space, avg_order_quantities_mip_1, width, label='MIP Product 1', color='g', alpha=0.5)



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Order Quantity')
# ax.set_title('Average Order Quantity vs Setup Costs Structure')
ax.set_xticks(x)
ax.set_xticklabels(costs)
ax.legend()
plt.ylim([0, 130])

fig.tight_layout()
plt.show()

# Repeat for Order Frequency and Average Total Cost
# setup costs structures
costs = ['Base Case', 'Major double', 'Major half', 'Minor Double', 'Minor Half']
x = np.arange(len(costs))  # the label locations

width = 0.10  # the width of the bars

fig, ax = plt.subplots()

# Create some additional space between RL and MIP
space = 0.05

rects1 = ax.bar(x - 2*width - space, order_freq_rl_0, width, label='RL Product 1', color='b')
rects2 = ax.bar(x - width - space, order_freq_rl_1, width, label='RL Product 2',color='b',alpha=0.65)
rects3 = ax.bar(x - space, joint_order_freq_rl, width, label='RL Joint',color='b',alpha=0.35)

rects4 = ax.bar(x + 2*space, order_freq_mip_0, width, label='MIP Product 1',color='g')
rects5 = ax.bar(x + width + 2*space, order_freq_mip_1, width, label='MIP Product 2', color='g',  alpha=0.65)
rects6 = ax.bar(x + 2*width + 2*space, joint_order_freq_mip, width, label='MIP Joint', color='g', alpha=0.35)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Order Frequency')
plt.ylim([0, 40])

# ax.set_title('Order Frequency vs Setup Costs Structure')
ax.set_xticks(x)
ax.set_xticklabels(costs)
ax.legend()

fig.tight_layout()
plt.show()

import matplotlib.pyplot as plt

product_1 = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
product_2 = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
time_steps = range(1, len(product_1) + 1)

product_1_orders = [t for t, ordered in zip(time_steps, product_1) if ordered]
product_2_orders = [t for t, ordered in zip(time_steps, product_2) if ordered]

plt.figure(figsize=(15, 3))

for i in time_steps:
    plt.axvline(x=i, linewidth=0.2, color='gray')

plt.scatter(product_1_orders, [0.7]*len(product_1_orders), color='b', label='Product 1', edgecolor='k')
plt.scatter(product_2_orders, [0.5]*len(product_2_orders), color='g', label='Product 2', edgecolor='k')

plt.yticks([0.7, 0.5], ['Product 1', 'Product 2'])
plt.xticks(time_steps)
plt.ylim(0.2, 1.8)  # Adjust y-axis limits

plt.xlabel('Time step')
plt.title('Order timeline for Products 1 and 2')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(False)
plt.box(False)

plt.show()

