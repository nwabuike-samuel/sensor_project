import os
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import *

class DataManager:
    def __init__(self, phase):
        self.phase = phase
        self.scaler_path = SCALER_FILES[phase]
        self.csv_file = CSV_FILE

    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "phase", "power", "thd", "shift", "voltage", "frequency", "anomaly"])
        else:
            # Load data into memory only if the file exists
            self.data = pd.read_csv(self.csv_file)

    def save_data(self, row):
        # Convert anomaly to a scalar if it's a NumPy array or list
        if isinstance(row[-1], (np.ndarray, list)):
            row[-1] = bool(row[-1][0])  # Convert to a Python boolean

        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        print(f"Saved data: {row}")

    def load_data(self):
        # Reload the data to ensure it's always up to date
        self.data = pd.read_csv(self.csv_file)
        return self.data

    def get_latest_data(self):
        if self.data is not None:
            # Reload the data if it wasn't loaded yet
            latest_data = self.data.tail(1).to_dict(orient="records")  # Convert the last row to a dictionary
            return latest_data
        else:
            print("Data is not loaded yet.")
            return {}

    def load_scaler(self):
        return joblib.load(self.scaler_path)
    
    def preprocess_data(self, data, phase, training=False):
        # Ensure `data` is a DataFrame
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame([data])  # Convert dictionary to DataFrame
            except Exception as e:
                raise ValueError(f"Error converting input data to DataFrame: {e}")
            
        phase_data = data[data['phase'] == phase].drop(columns=['phase', 'sensor', 'event', 'lag'])
        if training:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(phase_data)
            self.scalers[phase] = scaler
            joblib.dump(scaler, self.scaler_path)
        else:
            # scaler = joblib.load(f"scaler_phase_{phase}.pkl")
            # scaled_data = scaler.transform(phase_data)
            # Ensure phase-specific scaler is loaded
            scaler = self.load_scaler()
            scaled_data = scaler.transform(phase_data)

            # # Normalize data for the given phase
            # relevant_columns = ["power", "thd", "shift", "voltage", "frequency"]
            # filtered_data = data[relevant_columns]
            # scaled_data = scaler.transform([filtered_data])
        return scaled_data
