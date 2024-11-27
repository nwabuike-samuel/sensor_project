import pandas as pd

def json_to_csv(json_file, csv_file):
    df = pd.read_json(json_file, lines=True)
    df.to_csv(csv_file, index=False)
    print(f"Converted {json_file} to {csv_file}")

json_file = 'sensor_data.json'
csv_file = 'all_sensor_data.csv'

json_to_csv(json_file, csv_file)