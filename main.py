from scripts.datapreparation import load_data
from scripts.data_visualisation import plot_time_series
from scripts.forecasting_model import fit_arima, decompose_time_series, fit_ets, fit_prophet,compare_models

def main():
    # Load  the data
    data = load_data("IPG2211A2N.csv")

    # Plot  time series
    plot_time_series(data)

    decompose_time_series(data, "IPG2211A2N")

    # Fit ARIMA model
    arima_results = fit_arima(data, "IPG2211A2N", order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))

    # Fit ETS model
    ets_results = fit_ets(data, "IPG2211A2N", trend='add', seasonal='add', seasonal_periods=12)

    prophet_model, prophet_forecast = fit_prophet(data, "IPG2211A2N", yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    # Compare models
    compare_models(data, "IPG2211A2N")
if __name__ == "__main__":
    main()