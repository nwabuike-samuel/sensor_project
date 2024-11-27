from flask import Flask, render_template, jsonify
from flask_cors import CORS
import threading
import paho.mqtt.client as mqtt
from mqtt_handler import MQTTHandler
from config import *
import numpy as np

app = Flask(__name__)
CORS(app)

# autoencoder.load()
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

handler = MQTTHandler()

mqtt_client.on_message = handler.on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.subscribe(MQTT_TOPIC)
print(f"Subscribed to {MQTT_TOPIC}. Waiting for messages...")

# Start MQTT loop in a background thread
mqtt_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_mqtt')
def start_mqtt():
    global mqtt_thread
    if mqtt_thread is None or not mqtt_thread.is_alive():
        mqtt_thread = threading.Thread(target=mqtt_client.loop_start, daemon=True)
        mqtt_thread.start()
        return jsonify({"status": "success", "message": "MQTT connection started"})
    else:
        return jsonify({"status": "error", "message": "MQTT already running"})

@app.route('/stop_mqtt')
def stop_mqtt():
    mqtt_client.loop_stop()  # This stops the MQTT client loop
    return jsonify({"status": "success", "message": "MQTT connection stopped"})

@app.route('/get_data')
def get_data():
    try:
        latest_data = handler.get_latest_data()
        if latest_data:
            serialized_data = {key: (bool(value) if isinstance(value, np.bool_) else value)
                               for key, value in latest_data.items()}
            print("Data sent to client:", serialized_data)  # Log converted data
            return jsonify(serialized_data)
        else:
            print("No data available in MQTTHandler.")
            return jsonify({"message": "No data available."})
    except Exception as e:
        print(f"Error in get_data: {e}")
        return jsonify({"error": str(e)})

# @app.route('/get_data/<int:phase>')
# def get_data(phase):
#     try:
#         if phase not in PHASES:
#             return jsonify({"error": f"Invalid phase. Valid phases are {PHASES}."})

#         latest_data = handler.get_latest_data(phase)
#         if latest_data:
#             # Convert NumPy-specific types to standard Python types
#             serialized_data = {
#                 key: (bool(value) if isinstance(value, np.bool_) else value)
#                 for key, value in latest_data.items()
#             }
#             print(f"Phase {phase} data sent to client:", serialized_data)
#             return jsonify(serialized_data)
#         else:
#             print(f"No data available for phase {phase}.")
#             return jsonify({"message": f"No data available for phase {phase}."})
#     except Exception as e:
#         print(f"Error in get_data for phase {phase}: {e}")
#         return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
