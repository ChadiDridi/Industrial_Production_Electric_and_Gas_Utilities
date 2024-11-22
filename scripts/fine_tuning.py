import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def fine_tune_lstm(data, column, look_back=1):
    """Fine-tunes the LSTM model."""
    ts_data = data[column].dropna().values
    ts_data = ts_data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data = scaler.fit_transform(ts_data)

    train_size = int(len(ts_data) * 0.67)
    train, test = ts_data[0:train_size], ts_data[train_size:len(ts_data)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError(), MeanAbsoluteError()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(testX, testY), callbacks=[early_stopping], verbose=2)

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

    # Plot training & validation loss values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return model, trainScore, testScore

if __name__ == "__main__":
    from scripts.datapreparation import load_data
    data = load_data("../IPG2211A2N.csv")
    fine_tune_lstm(data, "data forecasting LSTM", look_back=1)