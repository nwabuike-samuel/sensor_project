import paho.mqtt.client as mqtt
import json

def process_message(payload):
    relevant_features = ["phase", "power", "thd", "shift", "voltage", "frequency"]
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
        with open("delta_sensor_data.json", "a") as file:
            json.dump(parsed_data, file)  # Append parsed data in JSON format
            file.write("\n")  # Add a newline for each entry

        print(f"Parsed Data: {parsed_data}")
    print("Data saved to sensor_data.json")


def on_message(client, userdata, message):
    # Decode the message payload
    payload = message.payload.decode('utf-8')
    # print(payload)
    process_message(payload)

def main():
    # MQTT client setup
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    # Connect to the broker
    broker = "mqtt.watter.co.uk"
    port = 1883
    topic = "aida/raw/997/#"
    
    client.connect(broker, port)
    client.subscribe(topic)

    print(f"Subscribed to {topic}. Waiting for messages...")
    
    # Start the loop to process incoming messages
    client.loop_forever()

if __name__ == "__main__":
    main()
