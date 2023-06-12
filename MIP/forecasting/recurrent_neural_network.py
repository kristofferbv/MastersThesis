import pandas as pd
import numpy as np
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Conv1D
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import tensorflow as tf

def normalize(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return (data - data_mean) / data_std, data_mean, data_std

def forecast(df, date,  n_time_periods, should_plot = True):
    # Split the data into training and testing sets
    df1 = df
    df = df.copy()
    # First, we decompose the series to get the seasonal component
    res = sm.tsa.seasonal_decompose(df["sales_quantity"], model='additive', period=52)
    # Then, we repeat the seasonal component for the desired number of periods
    trend_filled = res.trend.fillna(method='bfill').fillna(method="ffill")
    seasonal_filled = res.seasonal.fillna(method='ffill')

    # seasonal_data = np.tile(seasonal_filled, num_repetitions)[:num_periods]
    # data = seasonal_data
    train = df.loc[df.index <= date]["sales_quantity"].astype('float32')
    test = df.loc[df.index > date]["sales_quantity"].astype('float32')
    hei = test
    # train, train_mean, train_std = normalize(train)
    # test, test_mean, test_std = normalize(test)
    # train.index.freq = 'W-SUN'

    # Extract product_hash for model saving
    product_hash = df["product_hash"].iloc[0]

    # Prepare the data for input to the LSTM model
    train_values = np.reshape(train.values, (-1, 1))
    test_values = np.reshape(test.values, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))

    # Rescale both training and testing sets
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)
    n_steps = 26  # for yearly seasonality in weekly data; adjust as necessary
    n_features = 1  # for univariate time series data

    train_generator = TimeseriesGenerator(train_values, train_values, length=n_steps, batch_size=1)
    test_generator = TimeseriesGenerator(test_values, test_values, length=n_steps, batch_size=1)

    # Define the LSTM model



    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # inputs = layers.Input(shape=self.state_shape)
    #
    # # out = layers.LSTM(units=256, return_sequences=False, kernel_initializer="lecun_normal")(inputs)  # Adjust the number of units to your preference
    # out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(inputs)
    # out = layers.Dropout(rate=0.5)(out)
    # out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    # out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
    # out = layers.Dropout(rate=0.5)(out)
    # outputs = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)
    #
    # # Multiply with action upper bound
    # outputs = outputs * 200
    # model = tf.keras.Model(inputs, outputs)
    # return model



    # model= Sequential()
    # # model.add(SpatialDropout1D(0.3))
    # model = Sequential()
    # model.add(LSTM(128, activation='relu', input_shape=(n_steps, 1)))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, 1)))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    # model.compile(optimizer='adam', loss='mse')

    # Fit the LSTM model to the training data
    # model.fit(train_generator, epochs=200)
    model.fit(train_generator, epochs=20, validation_data=test_generator)

    # Make forecasts with the LSTM model
    forecast = []
    batch = np.reshape(train.values[-n_steps:],(-1, 1)).reshape((1, n_steps, 1))  # Scale the batch first
    for i in range(n_time_periods):
        yhat = model.predict(batch, verbose=0)  # yhat is in the scaled form
        forecast.append(yhat[0])
        batch = np.append(batch[:, 1:, :], [[yhat[0]]], axis=1)
    if should_plot:
        # Evaluate the forecast
        plt.plot(test.index[:n_time_periods], test.values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast, label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast recurrent')
        plt.legend()

        # Show the plot
        plt.show()
    # forecast = np.array(forecast).reshape(-1, 1)
    # forecast = scaler.inverse_transform(forecast)  # Inverse transform to get the original scale
    # Evaluate the forecast
    model.save(f'models/model_{product_hash}.h5')
    forecast = np.array(forecast)

    forecast[forecast < 0] = 0
    # forecast = forecast * train_std + train_mean
    mse = np.mean((hei.values[:n_time_periods] - forecast) ** 2)
    mae = np.mean(np.abs((test.values[:n_time_periods] - forecast)))
    smape = np.mean(200 * np.abs(test.values[:n_time_periods] - forecast) / (np.abs(test.values[:n_time_periods]) + np.abs(forecast)))
    std_dev = np.std(test.values[:n_time_periods] - forecast)  # Standard deviation of forecast errors
    cv = np.std(test.values[:n_time_periods] - forecast) / np.mean(test.values[:n_time_periods]) * 100
    rmse = np.sqrt(np.mean((test.values[:n_time_periods] - forecast) ** 2))
    print(f'MSE: {mse:.2f}')
    print("mae")
