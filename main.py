import threading
import time
import paho.mqtt.client as mqtt
from config import *
from data_manager import DataManager
from autoencoder import Autoencoder
from mqtt_handler import MQTTHandler
import pandas as pd
import os

# Initialize components
data_manager = DataManager(CSV_FILE)
autoencoder = Autoencoder(MODEL_FILE, SCALER_FILE, THRESHOLD)

# Initial training
if not os.path.exists(MODEL_FILE):
    initial_data = pd.read_csv(INITIAL_TRAINING_FILE)[["phase", "power", "thd", "shift", "voltage", "frequency"]].dropna().values
    autoencoder.train(initial_data, epochs=EPOCHS, batch_size=BATCH_SIZE, initial=True)
else:
    autoencoder.load()

# MQTT Setup
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
handler = MQTTHandler(client, data_manager, autoencoder, RETRAIN_TRIGGER)

client.on_message = handler.on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)

# Start MQTT loop in a separate thread
threading.Thread(target=client.loop_forever, daemon=True).start()

# Keep script running
while True:
    time.sleep(1)
