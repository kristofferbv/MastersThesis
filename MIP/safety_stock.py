import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import json
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split




# Initialize empty lists to store states and targets
states = []
targets = []

# Loop over each product index
for product_index in range(1):
    # Open the state and target files for this product and append the data to the states and targets lists
    with open(f"results/states{product_index}.txt", "r") as f:
        for line in f:
            states.append(json.loads(line))
    with open(f"results/targets{product_index}.txt", "r") as f:
        for line in f:
            targets.append(json.loads(line))

# Convert lists to numpy arrays for use with Keras

states = np.array(states)

targets = np.array(targets)
print(states.shape)
print(targets.shape)
states = states.reshape(-1, 4)
# Convert your states array into a DataFrame
states_df = pd.DataFrame(states, columns=['column1', 'column2', 'column3', 'column4'])

# Perform one-hot encoding on the last three columns
states_df = pd.get_dummies(states_df, columns=['column2', 'column3', 'column4'])

# Convert back to numpy array
states = states_df.values
# print(states.shape)
targets = np.array(targets)
targets = targets.reshape(-1)
print(targets.shape)
print(targets)
print(states)


encoder = OneHotEncoder(categories='auto')

# Select the last three columns and reshape them into a 2D array
to_encode = states[:, -3:]

# Fit the encoder and transform the data
one_hot_encoded = encoder.fit_transform(to_encode).toarray()

# Concatenate the original columns without the last three with the one-hot encoded columns
states = np.hstack([states[:, :-3], one_hot_encoded])

cb = CatBoostRegressor(n_estimators=50,
                       loss_function='RMSE',
                       learning_rate=0.1,
                       depth=10, task_type='CPU',
                       random_state=1,
                       verbose=False)


# Split your data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(states, targets, test_size=0.2, random_state=42)

# Initialize XGBRegressor
model = XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.0001, objective ='reg:squarederror')
# cb.fit(X_train, y_train,eval_set=[(X_valid, y_valid)], verbose=True)
# Fit model
# model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)

# Get predictions
# preds = cb.predict(X_valid)
# print(preds)

# Define a simple neural network model
# model = keras.Sequential([
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(200, activation='relu'),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(1)
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error',
#     metrics=[tf.keras.metrics.RootMeanSquaredError()])
#
# # # Train the model
# model.fit(states, targets, epochs=1000, validation_split=0.2)
# Define the base models
level0 = list()
level0.append(('xgb', XGBRegressor()))
level0.append(('lgbm', LGBMRegressor()))
level0.append(('cat', CatBoostRegressor(verbose=0)))  # turn off verbose for less output
level0.append(('rf', RandomForestRegressor()))


# Define meta learner model
level1 = LinearRegression()

# Define the stacking ensemble
stacking_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

# Fit the model on all available data
stacking_model.fit(X_train, y_train)
with open('stacking_model.pkl', 'wb') as file:
    pickle.dump(stacking_model, file)

y_pred = stacking_model.predict(X_valid)
# for i in range(len(y_pred)):
#     print(y_pred[i], y_valid[i])
# print("pred", y_pred)
# print("valid", y_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(rmse)
