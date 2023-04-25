import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def forecast(df, date):
    # Split the data into training and testing sets
    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]
    # train.index.freq = 'W-SUN'

    # Prepare the data for input to the LSTM model
    n_input = 7
    train_generator = TimeseriesGenerator(train.values, train.values, length=n_input, batch_size=1)
    test_generator = TimeseriesGenerator(test.values, test.values, length=n_input, batch_size=1)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # model.compile(optimizer='adam', loss='mse')

    # Fit the LSTM model to the training data
    model.fit(train_generator, epochs=200)

    # Make forecasts with the LSTM model
    forecast = []
    batch = train[-n_input:].values.reshape((1, n_input, 1))
    for i in range(len(test)):
        yhat = model.predict(batch, verbose=0)
        forecast.append(yhat[0])
        batch = np.append(batch[:,1:,:], yhat.reshape(1, 1, 1), axis=1)

    # Evaluate the forecast
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.title('Sales Forecast ARIMA')
    plt.legend()

    # Show the plot
    plt.show()

    # Evaluate the forecast
    mse = np.mean((test.values - forecast)**2)
    print(f'MSE: {mse:.2f}')
