Parameter:

- pH: The pH level of the water.
- Hardness: Water hardness, a measure of mineral content.
- Solids: Total dissolved solids in the water.
- Chloramines: Chloramines concentration in the water.
- Sulfate: Sulfate concentration in the water.
- Conductivity: Electrical conductivity of the water.
- Organic_carbon: Organic carbon content in the water.
- Trihalomethanes: Trihalomethanes concentration in the water.
- Turbidity: Turbidity level, a measure of water clarity.
- Potability: Target variable; indicates water potability with values 1 (potable) and 0 (not potable).

### Concept 1:

Training dataset when value missing: KNNImputer + Missing Flags
Production shrimp-monitoring app: 
Critical value missing
    -> request measurement again

Non-critical value missing
    -> estimate value + reduce confidence


### concept 2:

The Problem: Features Have Very Different Scales

pH            ≈ 7
Turbidity     ≈ 4
Hardness      ≈ 200
Solids        ≈ 22000

Without scaling ->

Suppose the network receives: [7.2, 200, 22000, 4]
The large feature dominates: 22000

The neural network may focus mostly on Solids simply because its numbers are much larger. Not because it's more important.

Hence use sklearn StandardScaler:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
self.features = scaler.fit_transform(self.features)

It transforms each column into: Mean = 0, Standard Deviation = 1


## Concept 3: "training a model" to "using a trained model as a product."

### Current state:

python3 train_waterdataset.py

the model:
1. Reads CSV
2. KNN Imputes
3. Scales
4. Trains for 20 epochs
5. Calculates accuracy

we do not want to retrain model every time

### Use save model forever:
Train Once
↓
Save Model
↓
Use Saved Model Forever


## Also save the scaling factor
- As user input value we need to know at what value we scaled.





Machine Learning Project: 
- Active Learning Loops
- Probabilistic Optimization (Strategies to explore vast)
- Multi-Objective Optimization Models
- Predictive Machine Learning Models
- Deep sequence structure function 
- Complex knowledge graphs
- Probabilistic models and algorithmic discoveries
- -> clear business impacts Risk assessments. Closed Loop models.

## What value consider good

## Drinking Water Reference Framework (Phase 1)

The AquaSkin AI Drinking Water Assessment uses a combination of:

- U.S. EPA National Primary Drinking Water Regulations
- U.S. EPA Secondary Drinking Water Standards
- WHO Guidelines for Drinking-water Quality
- Common water-quality industry classifications

> Note: Not all parameters have health-based limits. Some values are operational, aesthetic, or consumer-acceptability guidelines.

| Parameter | Recommended Range / Threshold | Primary Reference Source | Category |
|------------|------------|------------|------------|
| pH | 6.5 - 8.5 | EPA Secondary Drinking Water Standards | Operational |
| Hardness | < 180 mg/L (Very Hard > 180 mg/L) | Water Industry / USGS Classification | Comfort & Aesthetic |
| Total Dissolved Solids (TDS) | < 500 mg/L preferred | EPA Secondary Drinking Water Standards | Aesthetic |
| Chloramines | ≤ 4 mg/L | EPA Maximum Residual Disinfectant Level (MRDL) | Health-Based |
| Sulfate | < 250 mg/L | EPA Secondary Standard / WHO Guidance | Aesthetic |
| Conductivity | No universal health limit | Water Quality Indicator | Informational |
| Organic Carbon | No universal health limit | Water Quality Indicator | Informational |
| Trihalomethanes (TTHMs) | ≤ 80 µg/L | EPA Maximum Contaminant Level (MCL) | Health-Based |
| Turbidity | < 5 NTU (typically < 1 NTU preferred for treated water) | EPA / WHO Operational Guidance | Treatment Performance |

---

## Drinking Water Risk Categories

| Drinking Score | Classification |
|---|---|
| 90 - 100 | Excellent |
| 75 - 89 | Good |
| 50 - 74 | Caution |
| 25 - 49 | Poor |
| 0 - 24 | High Concern |

---


## Phase 2: Skin Comfort Assessment

The Skin Comfort Assessment evaluates water characteristics that may influence skin comfort, cleansing performance, and mineral residue buildup.

> These values are used for consumer comfort assessment and are not medical diagnostic thresholds.

| Parameter | Preferred Range | Reference Source | Rationale |
|------------|------------|------------|------------|
| pH | 6.5 - 8.5 | WHO Guidelines for Drinking-water Quality; EPA Secondary Drinking Water Standards | Extreme pH values may contribute to skin irritation and reduced comfort. |
| Hardness | < 120 mg/L Preferred <br> 120-180 mg/L Hard <br> >180 mg/L Very Hard | U.S. Geological Survey (USGS) Water Hardness Classification | Hard water is associated with soap residue, mineral deposits, and reduced skin comfort. |
| Total Dissolved Solids (TDS) | < 500 mg/L Preferred | EPA Secondary Drinking Water Regulations (SMCL) | High dissolved solids may contribute to mineral residue and unpleasant water feel. |
| Chloramines | ≤ 4 mg/L | EPA Maximum Residual Disinfectant Level (MRDL) | Elevated chloramines may contribute to dryness and irritation in sensitive individuals. |
| Sulfate | < 250 mg/L | EPA Secondary Drinking Water Standards; WHO Guidance | Elevated sulfate can affect consumer acceptability and water feel. |
| Conductivity | 50-1500 µS/cm Typical Drinking Water Range | Water Quality Monitoring Practice (Indicator Parameter) | Indicates dissolved ion concentration and mineralization level. |
| Organic Carbon | No Universal Consumer Limit | Water Treatment Industry Indicator Parameter | Used as an indicator of organic matter content. |
| Trihalomethanes (TTHMs) | ≤ 80 µg/L | EPA Maximum Contaminant Level (MCL) | Primarily evaluated for drinking-water safety rather than skin comfort. |
| Turbidity | < 5 NTU <br> < 1 NTU Preferred | WHO Drinking-water Guidelines; EPA Treatment Performance Goals | Higher turbidity may indicate reduced water quality and treatment effectiveness. |

---

## Hardness Classification (Skin Comfort)

| Hardness (mg/L as CaCO₃) | Classification | Reference |
|------------|------------|------------|
| 0 - 60 | Soft | USGS |
| 61 - 120 | Moderately Hard | USGS |
| 121 - 180 | Hard | USGS |
| >180 | Very Hard | USGS |

---

## Skin Comfort Score

| Score | Classification |
|------------|------------|
| 90 - 100 | Excellent |
| 75 - 89 | Good |
| 50 - 74 | Moderate |
| 25 - 49 | Poor |
| 0 - 24 | High Concern |

---

## Weighting Used in Skin Comfort Model

| Parameter | Weight |
|------------|------------|
| Hardness | 40% |
| Chloramines | 25% |
| Total Dissolved Solids | 20% |
| pH | 10% |
| Turbidity | 5% |

---

## References

1. World Health Organization (WHO). *Guidelines for Drinking-water Quality*, 4th Edition.
2. United States Environmental Protection Agency (EPA). *National Primary Drinking Water Regulations*.
3. United States Environmental Protection Agency (EPA). *Secondary Drinking Water Standards (SMCLs)*.
4. United States Geological Survey (USGS). *Water Hardness Classification*.
5. American Water Works Association (AWWA) guidance documents on water quality and treatment.





# Phase 3: Hair Comfort Assessment

The Hair Comfort Assessment evaluates water characteristics that may influence hair feel, manageability, mineral buildup, and washing performance.

> This assessment is informational only and does not diagnose hair loss, scalp disease, alopecia, dandruff, or any medical condition.

---

## Hair Comfort Reference Framework

| Parameter | Preferred Range | Reference Source | Rationale |
|------------|------------|------------|------------|
| pH | 6.5 - 8.5 | WHO Guidelines for Drinking-water Quality; EPA Secondary Drinking Water Standards | Extreme pH values may affect hair feel and scalp comfort. |
| Hardness | < 120 mg/L Preferred <br> 120-180 mg/L Hard <br> >180 mg/L Very Hard | U.S. Geological Survey (USGS) Water Hardness Classification | Hard water is often associated with mineral buildup on hair and reduced shampoo effectiveness. |
| Total Dissolved Solids (TDS) | < 500 mg/L Preferred | EPA Secondary Drinking Water Regulations (SMCL) | High mineral content may contribute to residue on hair. |
| Chloramines | ≤ 4 mg/L | EPA Maximum Residual Disinfectant Level (MRDL) | Elevated chloramines may contribute to dryness in sensitive individuals. |
| Sulfate | < 250 mg/L | EPA Secondary Drinking Water Standards; WHO Guidance | Elevated sulfate levels may negatively affect consumer acceptability. |
| Conductivity | 50-1500 µS/cm Typical Drinking Water Range | Water Quality Monitoring Practice | Indicator of dissolved minerals and ionic content. |
| Organic Carbon | No Universal Limit | Water Treatment Industry Indicator Parameter | Informational water-quality indicator. |
| Trihalomethanes (TTHMs) | ≤ 80 µg/L | EPA Maximum Contaminant Level (MCL) | Primarily a drinking-water parameter rather than a hair-comfort parameter. |
| Turbidity | < 5 NTU <br> < 1 NTU Preferred | WHO Drinking-water Guidelines; EPA Treatment Performance Goals | Higher turbidity may indicate reduced water quality conditions. |

---

## Hardness Classification (Hair Comfort)

| Hardness (mg/L as CaCO₃) | Classification | Reference |
|------------|------------|------------|
| 0 - 60 | Soft | USGS |
| 61 - 120 | Moderately Hard | USGS |
| 121 - 180 | Hard | USGS |
| >180 | Very Hard | USGS |

---

## Hair Comfort Score

| Score | Classification |
|------------|------------|
| 90 - 100 | Excellent |
| 75 - 89 | Good |
| 50 - 74 | Moderate |
| 25 - 49 | Poor |
| 0 - 24 | High Concern |

---

## Weighting Used in Hair Comfort Model

| Parameter | Weight |
|------------|------------|
| Hardness | 45% |
| Total Dissolved Solids (TDS) | 25% |
| Chloramines | 15% |
| pH | 10% |
| Turbidity | 5% |

---

## Hair Risk Factors

### High Hardness

Potential Effects:

- Mineral buildup on hair
- Reduced shampoo performance
- Duller appearance
- Increased residue after washing

Reference:
USGS Hardness Classification

---

### High Total Dissolved Solids (TDS)

Potential Effects:

- Residue accumulation
- Reduced clean-feeling after rinsing
- Increased mineral deposits

Reference:
EPA Secondary Drinking Water Standards

---

### Elevated Chloramines

Potential Effects:

- Dry feeling hair
- Reduced comfort for sensitive individuals

Reference:
EPA MRDL Guidelines

---

### Extreme pH

Potential Effects:

- Reduced scalp comfort
- Reduced hair feel and manageability

Reference:
WHO / EPA Operational Guidance

---

## Example Hair Comfort Algorithm

```python
hair_score = 100

if Hardness > 120:
    hair_score -= 30

if Solids > 500:
    hair_score -= 25

if Chloramines > 4:
    hair_score -= 15

if ph < 6.5 or ph > 8.5:
    hair_score -= 10

if Turbidity > 5:
    hair_score -= 5

hair_score = max(0, hair_score)
```

---

## Example Output

```text
═══════════════════════════
HAIR COMFORT REPORT
═══════════════════════════

Overall Hair Comfort Score:
38 / 100

Classification:
Poor

Primary Factors:

1. Very High Hardness
2. High Dissolved Solids
3. Elevated Chloramines

Possible Effects:

• Mineral buildup on hair
• Reduced shampoo performance
• Duller appearance
• Residue after washing

Recommendations:

✓ Monitor hardness levels
✓ Review water treatment options
✓ Check filtration performance
✓ Re-test after corrective actions

Assessment Confidence:
91%
```

---

## Hair Comfort Graphs

### Graph 1: Hair Impact Parameters

```text
Hardness
TDS
Chloramines
pH
Turbidity
```

Color Scheme:

- Green = Optimal
- Yellow = Warning
- Red = Concern

---

### Graph 2: Hair Comfort Gauge

```text
0------25------50------75------100

           ▲
          38
```

---

### Graph 3: Hair Risk Contribution

```text
Hardness       ███████████████ 45%
TDS            █████████ 25%
Chloramines    █████ 15%
pH             ███ 10%
Turbidity      █ 5%
```

---

## References

1. World Health Organization (WHO). *Guidelines for Drinking-water Quality*, 4th Edition.
2. United States Environmental Protection Agency (EPA). *National Primary Drinking Water Regulations*.
3. United States Environmental Protection Agency (EPA). *Secondary Drinking Water Standards (SMCLs)*.
4. United States Geological Survey (USGS). *Water Hardness Classification*.
5. American Water Works Association (AWWA) water-quality guidance documents.


## Test values 
1. Bad for Skin
7.5,350,2500,7,180,600,8,50,2

2. Bad for Hair
7.3,450,3500,3,200,700,8,40,1

3. Bad for Drinking
7.2,180,22000,8,350,1200,15,120,6

4. Good for Drinking
7.2,90,250,2,80,350,5,30,0.5

5. Good for Skin
7.1,60,350,1.5,80,300,5,30,0.5

6. Good for Hair
7.0,40,200,1,60,250,4,25,0.3



1. cloud infrastructures such as Azure, AWS, or


Step 1: Create requirements.txt
pip freeze > requirements.txt

Step 2: Create app.py
and check in local dashboard

Step 4: Connect Trained Model


Scenario,ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
Bad for Skin,7.5,350,2500,7,180,600,8,50,2
Bad for Hair,7.3,450,3500,3,200,700,8,40,1
Bad for Drinking,7.2,180,22000,8,350,1200,15,120,6
Good for Drinking,7.2,90,250,2,80,350,5,30,0.5
Good for Skin,7.1,60,350,1.5,80,300,5,30,0.5
Good for Hair,7.0,40,200,1,60,250,4,25,0.3


# Evaluation
✅ Training vs Validation Loss
✅ Confusion Matrix
✅ ROC Curve (with AUC)
✅ Precision-Recall Curve
✅ Feature Importance / SHAP Analysis