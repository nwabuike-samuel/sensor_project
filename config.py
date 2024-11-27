# Configuration file

# File paths
CSV_FILE = "sensor_data.csv"
INITIAL_TRAINING_FILE = "initial_data.csv"
SCALER_DIR = "scalers"  # Directory for storing phase-specific scaler files
MODEL_DIR = "models"    # Directory for storing phase-specific model files

# Phase-specific paths
PHASES = [0, 1, 2]
SCALER_FILES = {phase: f"{SCALER_DIR}/scaler_phase_{phase}.pkl" for phase in PHASES}
MODEL_FILES = {phase: f"{MODEL_DIR}/autoencoder_phase_{phase}.h5" for phase in PHASES}

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
# THRESHOLD = 0.01  # Anomaly detection threshold
RETRAIN_TRIGGER = 100  # Retrain model after this many new rows

# MQTT settings
MQTT_BROKER = "mqtt.watter.co.uk"
MQTT_PORT = 1883
MQTT_TOPIC = "aida/raw/997/#"
