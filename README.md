# Heart Disease Risk Prediction Web Application & API

An end-to-end Machine Learning deployment pipeline built with **Flask** and **Scikit-Learn**. This application dynamically loads a pre-trained **Logistic Regression** model along with its historical standard scaling constraints and encoded feature layout configurations. It is engineered with a hybrid entry point that supports **both** standard web browser form interactions and raw structured asynchronous JSON payloads from clients like Postman.

---

## 📝 Project Description

This project delivers a production-grade, asynchronous service designed to evaluate clinical heart disease risks using patient diagnostic metrics. Built on a lightweight **Flask microframework**, the backend decouples core analytical machine learning logic from data-science research environments (like Jupyter Notebooks) into a stable, deployable Python web application.

The underlying predictive core is driven by a highly calibrated **Logistic Regression model** selected for its high classification stability (~81.67% accuracy out-of-sample) and reliable probability distributions.

### 🛡️ Core Architectural Pillars & Preprocessing

Evaluating standalone patient observations introduces complex structural dependencies. When a single prediction request hits the server—whether from a web form or an API packet—it lacks the overall distribution context of the training dataset. This application seamlessly bridges that gap by running incoming data payloads through a 3-stage preprocessing matrix pipeline before model execution:

1. **One-Hot Encoding Expansion (`pd.get_dummies`):** Raw categorical entries (`cp`, `restecg`, `slope`, `thal`) are expanded into independent binary columns. This eliminates the false assumption that numerical categorical codes imply mathematical order, rank, or magnitude.
2. **Feature Dimensional Reindexing (`.reindex`):** Since an individual patient profile only holds one specific category code at a time, native encoding would collapse columns into an illegal dimension mismatch. The pipeline enforces a rigid **18-variable dimensional blueprint** captured during training, dynamically rebuilding unobserved column variables and filling them with `0`.
3. **Standard Scaling Normalization (`StandardScaler`):** Features are mapped across historical training standard deviations ($\sigma$) and means ($\mu$) stored inside the artifact pickle. This prevents numerical continuous metrics with wider ranges (such as `chol` or `thalach`) from disproportionately overpowering smaller, equally significant metrics.

---

## 📊 Dataset Feature Dictionary

The model consumes 13 patient clinical indicators to calculate the final target evaluation. Below is the semantic mapping for each raw field entry:

| Feature Key | Data Type | Value Mapping & Clinical Interpretations |
| :--- | :--- | :--- |
| **age** | Continuous | Patient age tracked in years. |
| **sex** | Binary | `1` = Male; `0` = Female. |
| **cp** | Categorical | **Chest Pain Type:**<br>• `0`: Typical Angina<br>• `1`: Atypical Angina<br>• `2`: Non-anginal Pain<br>• `3`: Asymptomatic |
| **trestbps** | Continuous | Resting blood pressure measured in mm Hg upon hospital admission. |
| **chol** | Continuous | Serum cholesterol levels measured in mg/dl. |
| **fbs** | Binary | **Fasting Blood Sugar > 120 mg/dl:**<br>• `1` = True;<br>• `0` = False. |
| **restecg** | Categorical | **Resting Electrocardiographic Results:**<br>• `0`: Normal<br>• `1`: ST-T Wave Abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV)<br>• `2`: Probable/definite left ventricular hypertrophy (Estes' criteria) |
| **thalach** | Continuous | Maximum heart rate achieved during clinical stress testing. |
| **exang** | Binary | **Exercise-Induced Angina:**<br>• `1` = Yes;<br>• `0` = No. |
| **oldpeak** | Continuous | ST depression induced by physical exercise relative to rest states. |
| **slope** | Categorical | **Slope of Peak Exercise ST Segment:**<br>• `0`: Upsloping<br>• `1`: Flat<br>• `2`: Downsloping |
| **ca** | Integer | Number of major coronary vessels (0–3) colored during a fluoroscopy procedure. |
| **thal** | Categorical | **Thalassemia Status:**<br>• `0`: Error / Missing Data Value<br>• `1`: Fixed Defect<br>• `2`: Normal Flow<br>• `3`: Reversible Defect |
| **target** | Binary | **The Prediction Label / Angiographic Disease Status:**<br>• `0` = No disease (< 50% diameter narrowing);<br>• `1` = Disease present (> 50% diameter narrowing). |

---

## 🧼 Data Quality & Cleansing Actions

During the exploratory data analysis (EDA) phase, cross-referencing with the official dataset documentation forum revealed systematic transcription anomalies across 7 specific data entries. The following remediation actions were systematically executed inside the preprocessing pipeline to safeguard model generalization boundaries:

* **Fluoroscopy Vessel Outliers (`ca`):** Entries **#93, 159, 164, 165, and 252** contained a value of `4`. In the original source clinical logs, these represent unobserved missing metrics (NaNs) and skew structural patterns.
* **Thalassemia Boundary Violations (`thal`):** Entries **#49 and 282** contained an invalid flag of `0` (which map directly to NaNs in the original clinical study).
* **Pipeline Action:** All 7 corrupted entries were explicitly dropped from the training corpus. This preserved structural health without injecting synthetic bias via arbitrary value imputations.

---

## 📂 Project Architecture

Ensure your local project directory structure is configured exactly as follows to allow Flask's template routing engine to find the UI layouts:

```text
HEART_DISEASE_RISK/
│
├── best_heart_disease_model.pkl    # Bundled Model, Scaler, and Feature blueprint
├── app.py                         # Hybrid Flask application server
├── README.md                       # Complete project documentation
└── templates/
    └── index.html                 # Frontend styled UI form template
