import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''
this script model_comparaison.py is for models performance and comparaison. The models aready exsists in forcasting_model.py
'''



def plot_model_performance(data, column, arima_results, ets_results, prophet_forecast, lstm_model, scaler, look_back=1):
    """Plots the performance of different models."""
    ts_data = data[column].dropna().values
    ts_data = ts_data.reshape(-1, 1)
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

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    lstm_train_predict = lstm_model.predict(trainX)
    lstm_test_predict = lstm_model.predict(testX)

    lstm_train_predict = scaler.inverse_transform(lstm_train_predict)
    lstm_test_predict = scaler.inverse_transform(lstm_test_predict)
    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])

    plt.figure(figsize=(14, 7))

    # Plot ARIMA results
    plt.plot(data.index, data[column], label='Original Data')
    plt.plot(data.index[-len(arima_results.fittedvalues):], arima_results.fittedvalues, label='ARIMA Fitted', color='red')

    # Plot ETS results
    plt.plot(data.index[-len(ets_results.fittedvalues):], ets_results.fittedvalues, label='ETS Fitted', color='green')

    # Plot Prophet results
    plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Forecast', color='blue')

    # Plot LSTM results
    plt.plot(data.index[look_back:len(lstm_train_predict) + look_back], lstm_train_predict, label='LSTM Train Predict', color='orange')
    plt.plot(data.index[len(train) + (look_back * 2) + 1:len(data) - 1], lstm_test_predict, label='LSTM Test Predict', color='purple')

    plt.title('Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def plot_model_errors(arima_results, ets_results, lstm_train_score, lstm_test_score):
    """Plots the errors of different models."""
    models = ['ARIMA', 'ETS', 'LSTM Train', 'LSTM Test']
    errors = [arima_results.aic, ets_results.aic, lstm_train_score, lstm_test_score]

    plt.figure(figsize=(10, 5))
    plt.bar(models, errors, color=['red', 'green', 'orange', 'purple'])
    plt.title('Model Errors')
    plt.xlabel('Models')
    plt.ylabel('Error (AIC/RMSE)')
    plt.show()