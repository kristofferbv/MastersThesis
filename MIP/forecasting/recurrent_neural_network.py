import pandas as pd
import numpy as np
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler


def forecast(df, date,  n_time_periods, should_plot = True):
    # Split the data into training and testing sets
    train = df.loc[df.index <= date]["sales_quantity"].astype('float32')
    test = df.loc[df.index > date]["sales_quantity"].astype('float32')
    # train.index.freq = 'W-SUN'

    # Extract product_hash for model saving
    product_hash = df["product_hash"].iloc[0]

    # Prepare the data for input to the LSTM model
    n_input = 10
    train_values = np.reshape(train.values, (-1, 1))
    test_values = np.reshape(test.values, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))

    # Rescale both training and testing sets
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)


    train_generator = TimeseriesGenerator(train_values, train_values, length=n_input, batch_size=1)
    test_generator = TimeseriesGenerator(test_values, test_values, length=n_input, batch_size=1)

    # Define the LSTM model

    model = Sequential()
    model.add(LSTM(50 ,activation='relu', input_shape=(n_input, 1)))
    # model.add(Dropout(0.2))
    # model.add(Dense(50, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    # model.compile(optimizer='adam', loss='mse')

    # Fit the LSTM model to the training data
    # model.fit(train_generator, epochs=200)
    model.fit(train_generator, epochs=50, validation_data=test_generator)

    # Make forecasts with the LSTM model
    forecast = []
    batch = np.reshape(train.values[-n_input:],(-1, 1)).reshape((1, n_input, 1))  # Scale the batch first
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
    mse = np.mean((test.values[:n_time_periods] - forecast) ** 2)
    mae = np.mean(np.abs((test.values[:n_time_periods] - forecast)))
    smape = np.mean(200 * np.abs(test.values[:n_time_periods] - forecast) / (np.abs(test.values[:n_time_periods]) + np.abs(forecast)))
    std_dev = np.std(test.values[:n_time_periods] - forecast)  # Standard deviation of forecast errors
    cv = np.std(test.values[:n_time_periods] - forecast) / np.mean(test.values[:n_time_periods]) * 100
    rmse = np.sqrt(np.mean((test.values[:n_time_periods] - forecast) ** 2))
    print(f'MSE: {mse:.2f}')
    print("mae")
