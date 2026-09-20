"""
Management Summary:
1. Load model
2. Load scaler
3. Accept water parameters
4. Scale inputs
5. Predict Potability
6. Generate Skin Score
7. Generate Hair Score
8. Create charts
9. Export report
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ==================================
# MODEL ARCHITECTURE
# ==================================

class WaterNet(nn.Module):

    def __init__(self, input_size, hidden_layers=[32,16,1]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ==================================
# LOAD SAVED FILES
# ==================================

scaler = joblib.load("scaler.pkl")

model = WaterNet(input_size=18, hidden_layers=[32,16,1])

model.load_state_dict(torch.load("water_model.pth"))

model.eval()


# ==================================
# WATER SAMPLE
# ==================================

water_sample = {

    "ph":7.8,
    "Hardness":320,
    "Solids":22000,
    "Chloramines":8,
    "Sulfate":350,
    "Conductivity":420,
    "Organic_carbon":12,
    "Trihalomethanes":80,
    "Turbidity":4,

    # Missing Flags
    "ph_missing":0,
    "Hardness_missing":0,
    "Solids_missing":0,
    "Chloramines_missing":0,
    "Sulfate_missing":0,
    "Conductivity_missing":0,
    "Organic_carbon_missing":0,
    "Trihalomethanes_missing":0,
    "Turbidity_missing":0
}


# ==================================
# PREDICT
# ==================================

X = np.array([list(water_sample.values())]) # it will take value from water_values 
X = scaler.transform(X)
X_tensor = torch.tensor(X,dtype=torch.float32)

with torch.no_grad():
    probability = model(X_tensor).item()

print(f"Potability Probability: {probability:.2%}")


# ==================================
# HAIR SCORE
# ==================================

hair_score = 100

if water_sample["Hardness"] > 250:
    hair_score -= 25

if water_sample["ph"] > 8:
    hair_score -= 10

if water_sample["Solids"] > 20000:
    hair_score -= 10

hair_score = max(0, hair_score)


# ==================================
# SKIN SCORE
# ==================================

skin_score = 100

if water_sample["Hardness"] > 250:
    skin_score -= 15

if water_sample["Turbidity"] > 5:
    skin_score -= 15

if water_sample["Chloramines"] > 7:
    skin_score -= 10

skin_score = max(0, skin_score)


# ==================================
# TEXT REPORT
# ==================================

print("\n========== AQUASKIN AI ==========")

print(f"Hair Comfort Score : {hair_score}/100")

print(f"Skin Comfort Score : {skin_score}/100")

print(f"Drinking Water Probability : {probability:.2%}")

print("\nPossible Concerns:")

if water_sample["Hardness"] > 250:
    print("- High hardness may contribute to mineral buildup.")

if water_sample["Chloramines"] > 7:
    print("- Elevated chloramines detected.")

if water_sample["Solids"] > 20000:
    print("- High dissolved solids detected.")

print("\nRecommendations")

print("- Monitor hardness regularly")
print("- Check filtration performance")
print("- Retest water periodically")


# ==================================
# GRAPH 1
# WATER PARAMETERS
# ==================================

parameters = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Turbidity"
]

values = [
    water_sample["ph"],
    water_sample["Hardness"],
    water_sample["Solids"] / 1000,
    water_sample["Chloramines"],
    water_sample["Turbidity"]
]

plt.figure(figsize=(8,5))

plt.bar(parameters, values)

plt.title(
    "Water Parameters Overview"
)

plt.savefig(
    "water_parameters.png"
)

plt.close()


# ==================================
# GRAPH 2
# SKIN HAIR SCORE
# ==================================

plt.figure(figsize=(6,5))

plt.bar(
    ["Hair","Skin"],
    [hair_score, skin_score],
    color=["blue","green"]
)

plt.ylim(0,100)

plt.title(
    "Skin & Hair Comfort Scores"
)

plt.savefig(
    "skin_hair_scores.png"
)

plt.close()


print(
    "\nGraphs saved successfully."
)