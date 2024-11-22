from scripts.datapreparation import load_data
from scripts.data_visualisation import plot_time_series
from scripts.forecasting_model import fit_arima,decompose_time_series

def main():
    # Load and preprocess the data
    data = load_data("IPG2211A2N.csv")

   
    plot_time_series(data)

 
    decompose_time_series(data, "IPG2211A2N")

 
    arima_results = fit_arima(data, "IPG2211A2N", order=(2, 1, 2))

if __name__ == "__main__":
    main()
