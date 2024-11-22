from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt

def fit_arima(data, column, order):
    """Fits an ARIMA model and plots forecasts."""
    model = ARIMA(data[column], order=order)
    results = model.fit()

    print(results.summary())
    return results

import matplotlib.pyplot as plt

def decompose_time_series(data, column):
    """Decomposes the time series and plots the components."""

    
    # Decompose the time series
    decomposition = seasonal_decompose(data, model='additive', period=12)
    
    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    plt.show()

if __name__ == "__main__":
    from datapreparation import load_data
    data = load_data("../IPG2211A2N.csv")
    arima_results = fit_arima(data, "IPG2211A2N", order=(2, 1, 2))
