import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(data):
    """Plots the time series."""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['IPG2211A2N'], label='Industrial Production: Electric and Gas Utilities')
    plt.title("Industrial Production Over Time")
    plt.xlabel("Date")
    plt.ylabel("Production")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rolling_statistics(data):
    """Plots rolling mean and standard deviation."""
    rolling_mean = data['IPG2211A2N'].rolling(window=12).mean()
    rolling_std = data['IPG2211A2N'].rolling(window=12).std()

    plt.figure(figsize=(14, 7))
    plt.plot(data['IPG2211A2N'], label='Original')
    plt.plot(rolling_mean, label='Rolling Mean (12 months)', color='red')
    plt.plot(rolling_std, label='Rolling Std (12 months)', color='black')
    plt.legend()
    plt.title("Rolling Mean & Standard Deviation")
    plt.show()

if __name__ == "__main__":
    from datapreparation import load_data
    data_path = "./IPG2211A2N.csv" 
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    plot_time_series(data)
    plot_rolling_statistics(data)