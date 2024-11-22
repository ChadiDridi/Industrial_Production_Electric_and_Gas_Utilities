import pandas as pd

def load_data(file_path):
    """Loads and preprocesses the dataset."""
    data = pd.read_csv(file_path)
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    return data



if __name__ == "__main__":
    data = load_data("../data/your_data.csv")
    print(data.head())
    print(data.info())
    print(data.describe())



