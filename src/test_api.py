import pandas as pd
import requests


DATA_FILE = "data/training/nids_training_dataset.csv"

df = pd.read_csv(DATA_FILE, nrows=1)

# Remove target
sample = df.drop(columns=["Target"])

# Convert to dictionary
features = sample.iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "features": features
    }
)

print("Status code:", response.status_code)

print("\nAPI response:")
print(response.json())