import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def decompose_time_series(data, column):
    """Decomposes the time series and plots the components."""
    ts_data = data[column].dropna()
    decomposition = seasonal_decompose(ts_data, model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    plt.show()

def fit_arima(data, column, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    """Fits an ARIMA model."""
    ts_data = data[column].dropna()
    model = ARIMA(ts_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    print(results.summary())
    return results

def fit_ets(data, column, trend='add', seasonal='add', seasonal_periods=12):
    """Fits an Exponential Smoothing (ETS) model."""
    ts_data = data[column].dropna()
    model = ExponentialSmoothing(ts_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    results = model.fit()
    print(results.summary())
    return results

def fit_prophet(data, column, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto'):
    """Fits a Prophet model."""
    ts_data = data[[column]].dropna().reset_index()
    ts_data.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
    model.fit(ts_data)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.show()
    return model, forecast

def fit_lstm(data, column, look_back=1):
    """Fits an LSTM model."""
    ts_data = data.reshape(-1, 1)  
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data = scaler.fit_transform(ts_data)

    train_size = int(len(ts_data) * 0.67)
    train, test = ts_data[0:train_size], ts_data[train_size:len(ts_data)]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = np.sqrt(np.mean((trainPredict[:, 0] - trainY[0]) ** 2))
    testScore = np.sqrt(np.mean((testPredict[:, 0] - testY[0]) ** 2))

    print(f'Train Score: {trainScore:.2f} RMSE')
    print(f'Test Score: {testScore:.2f} RMSE')

    return model, trainScore, testScore

def compare_models(data, column):
    """Compares different models and prints their performance."""
    print("Fitting ARIMA model")
    arima_results = fit_arima(data, column, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))

    print("Fitting ETS model")
    ets_results = fit_ets(data, column, trend='add', seasonal='add', seasonal_periods=12)

    print("Fitting Prophet model...")
    prophet_model, prophet_forecast = fit_prophet(data, column, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

    print("Fitting LSTM model")
    lstm_model, lstm_train_score, lstm_test_score = fit_lstm(data, column, look_back=1)

    print("Model comparison completed.")

if __name__ == "__main__":
    from datapreparation import load_data
    data = load_data("../IPG2211A2N.csv")
    
    # Decompose the time series
    decompose_time_series(data, "IPG2211A2N")
    
    # Compare 
    compare_models(data, "IPG2211A2N")