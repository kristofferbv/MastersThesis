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


# load demand data
with open('results/demand1.txt', 'r') as file:
    dict_demand = json.load(file)

# load actions data
with open('results/actions1.txt', 'r') as file:
    dict_action = json.load(file)

# load inventory data
with open('results/inventory1.txt', 'r') as file:
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
sequence_length =14  # 13 past demand levels + 1 current inventory level

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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        ) # The output dimension of the FFN should match the input embed_dim
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Set some hyperparameters
n_products = 2
n_features = 14
n_neurons = 200 # Number of neurons in the hidden layer, can be adjusted

# Define the model
# input_data = np.transpose(input_data, (0,2,1))
# model = get_actor(n_features, n_products)
print(input_data.shape)
model = Sequential([
    layers.LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(14, 2)),
    layers.Dropout(0.5),
    # TransformerBlock(embed_dim=100, num_heads=2, ff_dim=100), # embed_dim should match the output dimension of the previous layer
    # layers.Dropout(0.5),
    layers.LSTM(n_neurons, activation='relu'),
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
# model.summary()

# Train the model
# Replace `features` and `targets` with your actual data arrays
early_stopping = EarlyStopping(monitor='val_loss', patience=17)
print(input_data)

history = model.fit(input_data, target_data, batch_size=128, epochs=500, validation_split=0.2, callbacks=early_stopping)
model.save(f'actor_model')
# # Evaluate the model
# # Replace `test_features` and `test_targets` with your actual test data arrays
# test_loss = model.evaluate(test_features, test_targets)
#
#
#
#