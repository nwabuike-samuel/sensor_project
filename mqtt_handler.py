import json
from datetime import datetime
from config import *
from autoencoder import Autoencoder
from data_manager import DataManager

class MQTTHandler:
    def __init__(self):
        self.autoencoders = {phase: Autoencoder(phase) for phase in PHASES}
        self.data_managers = {phase: DataManager(phase) for phase in PHASES}
        self.latest_data = None  # Initialize latest_data
        self.retrain_trigger = RETRAIN_TRIGGER
        self.retrain_counter = 0  # Initialize retrain counter
        # self.threshold = THRESHOLD

        # Load models and scalers for each phase
        for phase, autoencoder in self.autoencoders.items():
            autoencoder.load()
        
    def parse_message(self, payload):
        # relevant_features = ["phase", "power", "thd", "shift", "voltage", "frequency"]
        if payload.startswith("raw"):
            # Split into header and data based on the first space
            header_part, data_part = payload.split(" ", 1)

            # Further split the header into individual fields (by commas)
            header_fields = header_part.split(",")

            # Initialize a dictionary to store parsed data
            parsed_data = {}

            # Parse header fields
            for item in header_fields[1:]:  # Skip the "raw" part
                if "=" in item:
                    key, value = item.split("=", 1)
                    try:
                        # Convert numeric values if possible
                        if value.endswith("i"):  # Integer values marked with 'i'
                            value = int(value[:-1])
                        else:
                            value = float(value)
                    except ValueError:
                        pass  # Keep as a string if conversion fails
                    # if key in relevant_features:
                    #     parsed_data[key] = value
                    parsed_data[key] = value

            # Parse data fields (after the first space)
            for item in data_part.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    try:
                        # Convert numeric values if possible
                        if value.endswith("i"):  # Integer values marked with 'i'
                            value = int(value[:-1])
                        else:
                            value = float(value)
                    except ValueError:
                        pass  # Keep as a string if conversion fails
                    # if key in relevant_features:
                    #     parsed_data[key] = value
                    parsed_data[key] = value

            # Save parsed data to a file
            with open("sensor_data.json", "a") as file:
                json.dump(parsed_data, file)  # Append parsed data in JSON format
                file.write("\n")  # Add a newline for each entry

            print(f"Parsed Data: {parsed_data}")
            print("Parsed Data saved to sensor_data.json")
            return parsed_data
        else:
            raise ValueError("Invalid payload format")
        
    def process_message(self, payload):
        try:
            # Parse the incoming payload
            parsed_data = self.parse_message(payload)

            # Extract relevant features
            relevant_features = [
                datetime.now().isoformat(),
                parsed_data.get("phase"),
                parsed_data.get("power"),
                parsed_data.get("thd"),
                parsed_data.get("shift"),
                parsed_data.get("voltage"),
                parsed_data.get("frequency"),
            ]

            print("Relevant Parsed data:", relevant_features)

            # Detect anomalies
            # features = relevant_features[1:]  # Exclude timestamp
            # is_anomaly = self.autoencoder.detect_anomaly(features)
            phase = parsed_data.get("phase")

            data_manager = self.data_managers[phase]
            autoencoder = self.autoencoders[phase]

            # Preprocess real-time data
            preprocessed_data = data_manager.preprocess_data(parsed_data, phase, training=False)

            # Detect anomalies
            is_anomaly = autoencoder.detect_anomaly(preprocessed_data)

            print("Detected anomaly:", is_anomaly)

            # Save data and anomaly to CSV
            relevant_features.append(is_anomaly)
            data_manager.save_data(relevant_features)
            self.retrain_counter += 1

            print("Relevant Parsed data and detected anomaly:", relevant_features)
            
            # Save latest data and anomaly status
            self.latest_data = {
                "timestamp": relevant_features[0],
                "phase": relevant_features[1],
                "power": relevant_features[2],
                "thd": relevant_features[3],
                "shift": relevant_features[4],
                "voltage": relevant_features[5],
                "frequency": relevant_features[6],
                "anomaly": bool(is_anomaly[0]),
            }

            print("Anomaly detected" if is_anomaly else "Data is nominal")

            print("Latest data updated:", self.latest_data)

            # Retrain model after reaching the retrain trigger
            if self.retrain_counter >= self.retrain_trigger:
                self.retrain_counter = 0
                for phase, autoencoder in self.autoencoders.items():
                    autoencoder.retrain()
        except ValueError as e:
            print(f"Error processing message: {e}")

    def on_message(self, client, userdata, msg):
        print("Received message from MQTT topic.")
        # Parse MQTT message
        payload = msg.payload.decode("utf-8")
        self.process_message(payload)
                
    def get_latest_data(self):
        # Return the latest data and anomaly status
        print("Returning latest data:", self.latest_data)
        return self.latest_data
