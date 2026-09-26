import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/predict"
DATA_FILE = "data/training/nids_training_dataset.csv"


# ============================================================
# LOAD SAMPLES
# ============================================================

print("=" * 70)
print("TESTING NIDS API")
print("=" * 70)

df = pd.read_csv(DATA_FILE, nrows=1000)


# ============================================================
# GET ONE BENIGN SAMPLE
# ============================================================

benign_sample = df[df["Target"] == 0].iloc[0]

benign_features = benign_sample.drop(
    labels=["Target"]
).to_dict()


# ============================================================
# GET ONE ATTACK SAMPLE
# ============================================================

attack_sample = df[df["Target"] == 1].iloc[0]

attack_features = attack_sample.drop(
    labels=["Target"]
).to_dict()


# ============================================================
# TEST BENIGN
# ============================================================

print("\n" + "=" * 70)
print("TEST 1: BENIGN TRAFFIC")
print("=" * 70)

response = requests.post(
    API_URL,
    json={
        "features": benign_features
    }
)

print("\nStatus code:", response.status_code)

print("\nAPI Response:")

print(response.json())


# ============================================================
# TEST ATTACK
# ============================================================

print("\n" + "=" * 70)
print("TEST 2: ATTACK TRAFFIC")
print("=" * 70)

response = requests.post(
    API_URL,
    json={
        "features": attack_features
    }
)

print("\nStatus code:", response.status_code)

print("\nAPI Response:")

print(response.json())


print("\n" + "=" * 70)
print("API TESTING COMPLETED")
print("=" * 70)