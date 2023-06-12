import json

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Conv2D
from keras import layers
from keras.optimizers import Adam, SGD
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import keras.backend as K

from keras.layers import Layer
class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
def get_actor(sequence, features):
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)

    inputs = layers.Input(shape=(sequence, features))

    # out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    # model.add(LSTM(128, activation='relu'))
    out = layers.GRU(units=200, return_sequences= True, activation="relu",kernel_initializer="lecun_normal")(inputs)  # Adjust the number of units to your preference
    out = layers.Dropout(rate=0.5)(out)
    out = layers.GRU(units=200, return_sequences= True, activation="relu",kernel_initializer="lecun_normal")(out)  # Adjust the number of units to your preference
    out = layers.Dropout(rate=0.5)(out)
    out = layers.GRU(units=200, return_sequences=False, activation="relu", kernel_initializer="lecun_normal")(out)  # Adjust the number of units to your preference
    out = layers.Dropout(rate=0.5)(out)
    # out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)
    # out = attention()(out)
    # out = layers.Flatten()(out)
    # out = layers.Dense(100, activation="selu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    # out = layers.Dense(400, activation="selu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    # out = layers.Dense(200, activation="selu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)


    # out = layers.Dense(200, activation="tanh", kernel_initializer="lecun_normal")(out)

    # out = layers.Dropout(rate=0.2)(out)

    # out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    outputs = layers.Dense(4)(out)
    outputs = outputs

    # Multiply with action upper bound
    # outputs = outputs * 100
    model = tf.keras.Model(inputs, outputs)
    return model

def feature_ma(state_input):
    demand = state_input[:, :, :, 0]  # Extract demand values from state_input
    demand_ma = np.mean(demand, axis=1, keepdims=True)  # Compute moving average along the time axis
    demand_ma = np.expand_dims(demand_ma, axis=2)  # Add an extra dimension to match the shape of state_input
    state_input_ma = np.concatenate((state_input, demand_ma), axis=3)  # Append moving average as a new feature
    return state_input_ma



def feature_tsla(state_input):
    demand = state_input[:, :, :, 0]  # Extract demand values from state_input
    peaks, _ = find_peaks(demand, prominence=1)  # Find peak indices for each product
    tsla = np.zeros_like(demand)  # Initialize time since last demand peak array
    for product_index in range(state_input.shape[2]):
        last_peak_index = -1
        for period in range(state_input.shape[1]):
            if period in peaks[product_index]:
                last_peak_index = period
            tsla[:, period, product_index] = period - last_peak_index  # Compute time since last peak
    state_input_tsla = np.concatenate((state_input, tsla[:, :, :, np.newaxis]), axis=3)  # Append time since last peak as a new feature
    return state_input_tsla


import numpy as np


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
    rate_of_change = np.diff(data, prepend=data[0]) / data
    return rate_of_change


def add_features(data, window_size, lag_steps):
    """Add new features to the given data."""
    # Compute the new features
    rolling_mean = compute_rolling_mean(data[:, 0], max(1, min(window_size, len(data))))
    lag_demand = compute_lag(data[:, 0], lag_steps)
    lag_inventory = compute_lag(data[:, 1], lag_steps)
    rate_of_change = compute_rate_of_change(data[:, 0])

    # Handle the size difference due to rolling mean
    feature_length = len(rolling_mean)
    lag_demand = lag_demand[-feature_length:]
    lag_inventory = lag_inventory[-feature_length:]
    rate_of_change = rate_of_change[-feature_length:]
    data = data[-feature_length:]
    # print("data")
    # print("rolling_mean", rolling_mean)
    # print("lag_dem", lag_demand)
    # print("lag inv", lag_inventory)
    # print("rate_of_change", rate_of_change)

    # Combine the original data with the new features
    data_extended = np.column_stack((data, rolling_mean, lag_demand, lag_inventory, rate_of_change))

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
sequence_length = 13  # 13 past demand levels + 1 current inventory level

# Initialize your data as 4D array
input_data = np.zeros((batch_size, sequence_length, n_products, 2))
for period in range(sequence_length, batch_size):
    # Iterate over each product
    for product_index in range(n_products):
        # Get the demand values for this product for the last 13 periods
        demand_features = dict_demand[str(product_index)][period-int(sequence_length):period]
        print(len(dict_demand["0"]))
        # Pad with zeros if not enough periods
        if len(demand_features) < sequence_length:
            demand_features = [0]*(sequence_length-len(demand_features)) + demand_features
        # Get the inventory level for this product for the current period
        inventory_features = [dict_inventory[str(i)][product_index] for i in range(period - sequence_length + 1, period + 1)]
        # Pad with zeros if not enough periods
        if len(inventory_features) < sequence_length:
            inventory_features = [0]*(sequence_length-len(inventory_features)) + inventory_features
        # Pair demand and inventory features
        paired_features = [[d, i] for d, i in zip(demand_features, inventory_features)]
        # Store in the input_data array
        input_data[period-sequence_length+1, :sequence_length, product_index, :] = paired_features

    # Get the order quantities for this period
    order_quantities = dict_action[str(period-int(sequence_length)+2)]
    # Store in the target array
    target_data[period-sequence_length+1, :] = order_quantities
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
#



# Set some hyperparameters
n_products = 4
n_features = 13
n_neurons = 100 # Number of neurons in the hidden layer, can be adjusted
print(input_data.shape)
print(len(input_data.shape))
# input_data = input_data.reshape(13, -1)

print(input_data.shape)
# input_data = feature_ma(input_data)
# input_data = feature_tslp(input_data)
print(input_data)

# Feature engineering
window_size = 3
lag_steps = 1
num_features_original = 2
num_features_new = 6  # update this based on the number of new features added

# Create a new array to hold the data with extra features
data_extended = np.zeros((input_data.shape[0], input_data.shape[1] - window_size + 1, input_data.shape[2], num_features_new))

# for batch_id in range(input_data.shape[0]):
#     for product_id in range(input_data.shape[2]):
#         sequence_data = input_data[batch_id, :, product_id, :]
#         sequence_data_extended = add_features(sequence_data, window_size, lag_steps)
#         data_extended[batch_id, :len(sequence_data_extended), product_id, :] = sequence_data_extended
#
# input_data = data_extended
# print(input_data)
# input_data= input_data.reshape(1325, 11, -1)



# Define the model
# input_data = np.transpose(input_data, (0,2,1))
model = get_actor(n_features, n_products)
model = Sequential([
    layers.GRU(n_neurons, activation='relu', return_sequences=True, input_shape=(11, n_products*6)),
    layers.Dropout(0.5),
    # layers.LSTM(n_neurons, activation='relu', return_sequences=True),
    # layers.Dropout(0.5),
    layers.GRU(n_neurons, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_products)
])


#get_actor(n_products, n_features) #Sequential([
#     Flatten(input_shape=(n_products, n_features)),  # Flatten the input
#     Dense(n_neurons, activation='relu'),
#     layers.Dropout(0.1),
#     Dense(n_neurons, activation='relu'),
#     layers.Dropout(0.1),# Hidden layer
#     Dense(n_products)  # Output layer
# ])#

# Compile the model
optimizer = Adam(learning_rate=0.001)  # You can replace 0.001 with your desired learning rate
model.compile(optimizer=optimizer, loss='mae')  # Use mean squared error as the loss function

# Print a summary of the model
model.summary()

# Train the model
# Replace `features` and `targets` with your actual data arrays
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(input_data, target_data, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
model.save(f'actor_model')
# # Evaluate the model
# # Replace `test_features` and `test_targets` with your actual test data arrays
# test_loss = model.evaluate(test_features, test_targets)
#
#
#
#