import numpy as np
import joblib
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

    def train_and_save(self, data_manager, data, phases):
        for phase in phases:
            phase_data = data_manager.preprocess_data(data, phase, training=True)
            autoencoder = self.create_model()
            autoencoder.fit(phase_data, phase_data, epochs=10, batch_size=32)
            autoencoder.save(f"autoencoder_phase_{phase}.h5")
            self.models[phase] = autoencoder
    
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
