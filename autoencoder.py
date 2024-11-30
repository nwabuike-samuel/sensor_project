import pandas as pd 
import numpy as np
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
from config import *


class Autoencoder:
    def __init__(self, phase):
        self.phase = phase
        self.model_path = MODEL_FILES[phase]
        self.scaler_path = SCALER_FILES[phase]
        self.model = None
        self.scaler = None
        # self.threshold = THRESHOLD

    def create_model(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(128, activation="relu", kernel_initializer=HeNormal(), bias_initializer='zeros')(input_layer)
        encoder = Dense(64, activation='relu', kernel_initializer=HeNormal(), bias_initializer='ones')(encoder)
        encoder = Dense(32, activation='relu', kernel_initializer=HeNormal(), bias_initializer='ones')(encoder)
        encoder = Dense(16, activation='relu', kernel_initializer=HeNormal(), bias_initializer='ones')(encoder)

        encoder = Dense(2, activation="relu", kernel_initializer=HeNormal(), bias_initializer='ones')(encoder)

        decoder = Dense(16, activation='relu', kernel_initializer=HeNormal(), bias_initializer='ones')(encoder)
        decoder = Dense(32, activation="relu", kernel_initializer=HeNormal(), bias_initializer='ones')(decoder)
        decoder = Dense(64, activation="relu", kernel_initializer=HeNormal(), bias_initializer='ones')(decoder)
        decoder = Dense(128, activation="relu", kernel_initializer=HeNormal(), bias_initializer='ones')(decoder)
        decoder = Dense(input_dim, activation="sigmoid", kernel_initializer=HeNormal(), bias_initializer='zeros')(decoder)
        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        
        return model

    # def train(self, data, epochs, batch_size, initial=False):
        # Normalize data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)

        # Create or load model
        if initial or not os.path.exists(self.model_file):
            self.model = self.create_model(input_dim=scaled_data.shape[1])
        else:
            self.model = load_model(self.model_file)

        # Train model and scaler
        self.model.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, shuffle=True)

        # Save model
        save_model(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)
        print("Model and scaler trained and saved.")

    # def train_and_save(self, data_manager, data, phases):
    #     for phase in phases:
    #         phase_data = data_manager.preprocess_data(data, phase, training=True)
    #         autoencoder = self.create_model()
    #         autoencoder.fit(phase_data, phase_data, epochs=10, batch_size=32)
    #         autoencoder.save(f"autoencoder_phase_{phase}.h5")
    #         self.models[phase] = autoencoder

    def train(self, data, epochs, batch_size, initial=False):
        # Normalize data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)

        # Create or load model
        if initial or not os.path.exists(self.model_path):
            self.model = self.create_model(input_dim=scaled_data.shape[1])
        else:
            self.model = load_model(self.model_path)

        # Train the model
        self.model.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, shuffle=True)

        # Save updated model and scaler
        save_model(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("Model and scaler trained and saved.")
    
    def load(self):
        # Load model and scaler
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        print("Model and scaler loaded successfully.")

    # def load_model(self, phase):
    #     return load_model(f"autoencoder_phase_{phase}.h5")
    
    # def detect_anomaly(self, data_row):
    #     # Normalize incoming data
    #     scaled_data = self.scaler.transform([data_row])

    #     # Predict reconstruction
    #     reconstruction = self.model.predict(scaled_data)
    #     error = np.mean((scaled_data - reconstruction) ** 2)
    #     return error > self.threshold

    # def detect_anomalies(self, real_time_data, phase):
    #     # Preprocess using the saved scaler
    #     scaled_data = self.data_manager.preprocess_data(real_time_data, phase)

    #     # Load the phase-specific model
    #     model = self.autoencoder_manager.load_model(phase)

    #     # Predict reconstruction
    #     reconstructions = model.predict(scaled_data)
    #     reconstruction_error = np.mean(np.square(scaled_data - reconstructions), axis=1)

        ## Detect anomalies
        # return reconstruction_error > self.threshold

    def detect_anomaly(self, data):
        # Get reconstruction error
        reconstructed_data = self.model.predict(data)
        reconstruction_error = ((data - reconstructed_data) ** 2).mean(axis=1)
        # Return True if the error exceeds a threshold (set your threshold)
        # threshold = 0.1  # Example threshold
        threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)
        return reconstruction_error > threshold

    def retrain(self):
        try:
            # Load entire JSON data
            json_file_path = "sensor_data.json"  # Path to your JSON file
            all_data = []
            with open(json_file_path, mode="r") as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        all_data.append(json.loads(line))  # Parse each line as JSON
            
            if not all_data:
                print("No data available in JSON file for retraining.")
                return

            # Convert to DataFrame for preprocessing
            data_df = pd.DataFrame(all_data)
            data_df = data_df.drop(columns=["sensor", "event", "lag"])

             # Extract data for the specified phase
            phase_data = data_df[data_df["phase"] == self.phase].drop(columns=['phase'])

            if phase_data.empty:
                print(f"No data available for phase {self.phase}. Skipping retrain.")
                return

            # Drop irrelevant columns based on your original notebook preprocessing
            features = phase_data.dropna()

            # Ensure there is enough data for training
            if features.empty or len(features) < 2:
                print(f"Insufficient data for phase {self.phase} retraining.")
                return

            # Retrain the model
            print(f"Retraining model for Phase {self.phase}...")
            self.train(features, epochs=EPOCHS, batch_size=BATCH_SIZE, initial=True)

            print(f"Retrained and updated model and scaler for phase {self.phase}.")

        except FileNotFoundError:
            print(f"Error: JSON file {json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
        except Exception as e:
            print(f"Error during retraining: {e}")