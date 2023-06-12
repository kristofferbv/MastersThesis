import json

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Conv2D
from keras import layers
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
import keras.backend as K

from keras.layers import Layer

def compute_rolling_mean(data, window_size):
    """Compute the rolling mean for the given data."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def compute_lag(data, lag_steps):
    """Compute a lag feature for the given data."""
    lag_feature = np.roll(data, shift=lag_steps)

    lag_feature[:lag_steps] = 0  # handle edge case for the initial elements
    return lag_feature


def compute_rate_of_change(data):
    """Compute the rate of change for the given data."""
    rate_of_change = np.zeros_like(data)
    non_zero_indices = np.nonzero(data)
    rate_of_change[non_zero_indices] = np.diff(data[non_zero_indices], prepend=data[non_zero_indices][0]) / data[non_zero_indices]
    return rate_of_change
# Define a function to compute sine and cosine transformations
def compute_cyclic_feature(value, max_value):
    """Compute sine and cosine transformations for a cyclic feature."""
    value_scaled = (value - 1) / max_value  # Shift indices to start from 0
    value_sin = np.sin(2 * np.pi * value_scaled)
    value_cos = np.cos(2 * np.pi * value_scaled)
    return value_sin, value_cos


def add_features(data, window_size, lag_steps):
    # Compute the new features
    rolling_mean = compute_rolling_mean(data, window_size)
    lag_demand = compute_lag(data, lag_steps)
    rate_of_change = compute_rate_of_change(data)

    # Handle the size difference due to rolling mean
    feature_length = len(rolling_mean)
    lag_demand = lag_demand[-feature_length:]
    rate_of_change = rate_of_change[-feature_length:]
    data = data[-feature_length:]

    # Combine the original data with the new features
    data_extended = np.column_stack((data, rolling_mean))

    # Replace any nan or inf values with a large negative number
    data_extended = np.nan_to_num(data_extended, nan=-1e10, posinf=-1e10, neginf=-1e10)

    return data_extended
# load demand data
with open('results/demand.txt', 'r') as file:
    dict_demand = json.load(file)

# load actions data
with open('results/actions.txt', 'r') as file:
    dict_action = json.load(file)

# load inventory data
with open('results/inventory.txt', 'r') as file:
    dict_inventory = json.load(file)

n_products = len(dict_demand)
n_features = 14  # 5 demand values + 5 inventory values
batch_size = len(dict_inventory)

# Initialize the input array
input_data = np.zeros((batch_size, n_products, n_features))

# Initialize the target array
target_data = np.zeros((batch_size, n_products))
# Loop over all products
# Iterate over each time period
sequence_length = 14  # 13 past demand levels + 1 current inventory level

# Initialize the input array
input_data = np.zeros((batch_size, sequence_length, n_products))

for period in range(sequence_length-1, batch_size):
    # Iterate over each product
    for product_index in range(n_products):
        # Get the demand values for this product for the last 13 periods
        print("lengde!" , len(dict_demand[str(product_index)]))
        demand_features = dict_demand[str(product_index)][period-sequence_length+1:period]
        # Pad with zeros if not enough periods
        if len(demand_features) < 13:
            demand_features = [0]*(13-len(demand_features)) + demand_features
        # Get the inventory level for this product for the current period
        inventory_features = [dict_inventory[str(period-2)][product_index]]
        # Concatenate demand and inventory features
        input_data[period-sequence_length+1, :, product_index] = demand_features + inventory_features

    # Get the order quantities for this period
    order_quantities = dict_action[str(period-sequence_length+2)]
    # Store in the target array
    target_data[period-sequence_length+1, :] = order_quantities
print(input_data.shape)
input_data = input_data[:11690, :, :]

# print(input_data.shape)
# input_data = feature_ma(input_data)
# input_data = feature_tslp(input_data)
# print(input_data.shape)
# print(input_data.shape)
# print(input_data)
# print(target_data.shape)
# print(target_data)

# Saving arrays to file
# np.save('features234.npy', input_data)  # Replace 'features' and 'targets' with your actual arrays
# np.save('targets234.npy', target_data)
# input_data = np.load('features234.npy')  # Replace 'features' and 'targets' with your actual arrays
# target_data = np.load('targets234.npy')

# original_shape = input_data.shape
# # reshape it into 2D array
# input_data_2D = np.reshape(input_data, (-1, original_shape[-1]))
#
# scaler = StandardScaler()
# input_data_scaled_2D = scaler.fit_transform(input_data_2D)
#
# # reshape it back into 3D
# input_data_ = np.reshape(input_data_scaled_2D, original_shape)

window_size = 3
lag_steps = 1
num_features_original = 1  # just demand
num_features_new = 2  # demand + rolling_mean + lag_demand + rate_of_change

# Create a new array to hold the data with extra features
data_extended = np.zeros((input_data.shape[0], input_data.shape[1] - window_size + 1, input_data.shape[2], num_features_new))
print(input_data[-20,:,:])
for batch_id in range(input_data.shape[0]):
    for product_id in range(input_data.shape[2]):
        sequence_data = input_data[batch_id, :, product_id]
        sequence_data_extended = add_features(sequence_data, window_size, lag_steps)
        data_extended[batch_id, :len(sequence_data_extended), product_id, :] = sequence_data_extended

input_data = data_extended
# print(input_data.shape)
# print(input_data)

# Set some hyperparameters
n_products = 4
n_features = 14
n_neurons = 100 # Number of neurons in the hidden layer, can be adjusted

# Define the model
# input_data = np.transpose(input_data, (0,2,1))
# model = get_actor(n_features, n_products)
week_numbers = np.array([(period // 7) % 52 + 1 for period in range(11690)])  # Assuming each week consists of 7 periods and each season is 52 weeks
week_sin, week_cos = compute_cyclic_feature(week_numbers, 52)
input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], -1)  # shape: (batch_size, sequence_length, n_products*n_features)

# Assume that num_samples, sequence_length, and n_products are correctly defined somewhere earlier in your code
num_samples = len(week_sin)
sequence_length = 12
n_products = 4

# Reshape week_sin and week_cos into shape (num_samples, sequence_length, 1)
week_sin_reshaped = np.repeat(week_sin.reshape(-1, 1), repeats=sequence_length, axis=-1).reshape(num_samples, sequence_length, 1)
week_cos_reshaped = np.repeat(week_cos.reshape(-1, 1), repeats=sequence_length, axis=-1).reshape(num_samples, sequence_length, 1)

# Repeat along the last dimension to match the number of products
week_sin_repeated = np.repeat(week_sin_reshaped, repeats=n_products, axis=-1)
week_cos_repeated = np.repeat(week_cos_reshaped, repeats=n_products, axis=-1)

# Now, you can append these new arrays to input_data
input_data = np.append(input_data, week_sin_repeated, axis=-1)
input_data = np.append(input_data, week_cos_repeated, axis=-1)





model = Sequential([
    layers.GRU(n_neurons, activation='relu', return_sequences=True, input_shape=(12, 16)),
    layers.Dropout(0.8),
    layers.GRU(n_neurons, activation='relu'),
    layers.Dropout(0.8),
    layers.Dense(n_products)
])


# Compile the model
optimizer = Adam(learning_rate=0.001)  # You can replace 0.001 with your desired learning rate
model.compile(optimizer=optimizer, loss='mae')  # Use mean squared error as the loss function

# Print a summary of the model
model.summary()

# Train the model
# Replace `features` and `targets` with your actual data arrays
early_stopping = EarlyStopping(monitor='val_loss', patience=30)

history = model.fit(input_data, target_data, batch_size=264, epochs=500, validation_split=0.2)
model.save(f'actor_model')
# # Evaluate the model
# # Replace `test_features` and `test_targets` with your actual test data arrays
# test_loss = model.evaluate(test_features, test_targets)
#
#
#
