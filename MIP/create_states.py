import json

import numpy as np

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras import layers
from keras.optimizers import Adam, SGD


def get_actor(sequence, features):
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)

    inputs = layers.Input(shape=(sequence, features))
    # out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    # model.add(LSTM(128, activation='relu'))
    out = layers.LSTM(units=32, return_sequences= True, activation="sigmoid",kernel_initializer="lecun_normal")(inputs)  # Adjust the number of units to your preference
    # attention_layer = attention()(out)
    out = layers.Flatten()(out)
    out = layers.Dense(100, activation="relu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.1)(out)
    out = layers.Dense(100, activation="relu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.1)(out)


    # out = layers.Dense(200, activation="tanh", kernel_initializer="lecun_normal")(out)

    # out = layers.Dropout(rate=0.2)(out)

    # out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    outputs = layers.Dense(4)(out)

    # Multiply with action upper bound
    # outputs = outputs * 100
    model = tf.keras.Model(inputs, outputs)
    return model

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
n_features = 24  # 5 demand values + 5 inventory values
batch_size = len(dict_inventory)

# Initialize the input array
input_data = np.zeros((batch_size, n_products, n_features))
# # Initialize the target array
target_data = np.zeros((batch_size, n_products))
# Loop over all products
# Iterate over each time period
for period in range(12, batch_size):
    # Iterate over each product
    for product_index in range(n_products):
        # Get the last 5 demand values for this product up to the current period
        demand_features = dict_demand[str(product_index)][:period+1][-12:]
        # Pad with zeros if not enough periods
        demand_features = [0]*(12-len(demand_features)) + demand_features
        # Get the last 5 inventory levels for this product up to the current period
        inventory_features = [dict_inventory[str(i)][product_index] for i in range(period+1)][-12:]
        # Pad with zeros if not enough periods
        inventory_features = [0]*(12-len(inventory_features)) + inventory_features
        # Concatenate demand and inventory features
        input_data[period, product_index, :] = demand_features + inventory_features


    # Get the order quantities for this period
    order_quantities = dict_action[str(period-11)]
    # Store in the target array
    target_data[period, :] = order_quantities

print(input_data.shape)
# Saving arrays to file
# np.save('features.npy', input_data)  # Replace 'features' and 'targets' with your actual arrays
# np.save('targets.npy', target_data)
# input_data = np.load('features.npy')  # Replace 'features' and 'targets' with your actual arrays
# target_data = np.load('targets.npy')
#
#
# print(input_data.shape)
# print(target_data.shape)
#
#
# # Set some hyperparameters
# n_products = 4
# n_features = 24
# n_neurons = 100  # Number of neurons in the hidden layer, can be adjusted
#
# # Define the model
# input_data = np.transpose(input_data, (0,2,1))
# model = get_actor(n_products, n_features) #Sequential([
# #     Flatten(input_shape=(n_products, n_features)),  # Flatten the input
# #     Dense(n_neurons, activation='relu'),
# #     layers.Dropout(0.1),
# #     Dense(n_neurons, activation='relu'),
# #     layers.Dropout(0.1),# Hidden layer
# #     Dense(n_products)  # Output layer
# # ])#
#
# # Compile the model
# optimizer = Adam(learning_rate=0.0001)  # You can replace 0.001 with your desired learning rate
# model.compile(optimizer=optimizer, loss='mse')  # Use mean squared error as the loss function
#
# # Print a summary of the model
# model.summary()
#
# # Train the model
# # Replace `features` and `targets` with your actual data arrays
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#
# history = model.fit(input_data, target_data, epochs=1000, validation_split=0.2)
#
# # # Evaluate the model
# # # Replace `test_features` and `test_targets` with your actual test data arrays
# # test_loss = model.evaluate(test_features, test_targets)
#
#
#
#