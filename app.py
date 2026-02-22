import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# -----------------------------
# Load model & scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model-2.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "standard_scaler.joblib"))

# All model features (internal)
FEATURE_COLUMNS = [
    "PackVoltage_V",
    "CellVoltage_V",
    "DemandVoltage_V",
    "ChargeCurrent_A",
    "DemandCurrent_A",
    "SOC_%",
    "MaxTemp_C",
    "MinTemp_C",
    "AvgTemp_C",
    "AmbientTemp_C",
    "InternalResistance_mOhm",
    "StateOfHealth_%",
    "VibrationLevel_mg",
    "MoistureDetected",
    "ChargePower_kW",
    "Pressure_kPa",
    "ChargingStage_Handshake",
    "ChargingStage_Parameter_Config",
    "ChargingStage_Recharge",
    "BMS_Status_OK",
    "BMS_Status_Warning"
]

# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return "EV Battery Thermal Runaway Prediction API is running."

# -----------------------------
# API endpoint (unchanged)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)

    return jsonify({
        "prediction": int(pred[0]),
        "safe_probability": float(prob[0][0]),
        "risk_probability": float(prob[0][1])
    })

# -----------------------------
# SIMPLE BROWSER UI
# -----------------------------
@app.route("/ui", methods=["GET", "POST"])
def ui():
    result = None

    if request.method == "POST":
        # Only key fields from UI
        input_data = {
            "PackVoltage_V": float(request.form["PackVoltage_V"]),
            "ChargeCurrent_A": float(request.form["ChargeCurrent_A"]),
            "SOC_%": float(request.form["SOC_%"]),
            "MaxTemp_C": float(request.form["MaxTemp_C"]),
            "AmbientTemp_C": float(request.form["AmbientTemp_C"]),
            "MoistureDetected": int(request.form["MoistureDetected"])
        }

        # Fill remaining features with 0
        for col in FEATURE_COLUMNS:
            input_data.setdefault(col, 0)

        df = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)

        result = {
            "label": "SAFE" if pred[0] == 0 else "THERMAL RUNAWAY RISK",
            "safe_prob": round(prob[0][0], 3),
            "risk_prob": round(prob[0][1], 3)
        }

    return render_template_string("""
    <html>
    <head>
        <title>EV Battery Thermal Prediction</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            input, select { width: 220px; padding: 6px; margin-bottom: 10px; }
            button { padding: 10px 20px; }
            .safe { color: green; font-weight: bold; }
            .risk { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>EV Battery Thermal Runaway Predictor</h2>

        <form method="post">
            <label>Pack Voltage (V)</label><br>
            <input type="number" step="any" name="PackVoltage_V" required><br>

            <label>Charge Current (A)</label><br>
            <input type="number" step="any" name="ChargeCurrent_A" required><br>

            <label>State of Charge (%)</label><br>
            <input type="number" step="any" name="SOC_%" required><br>

            <label>Max Temperature (°C)</label><br>
            <input type="number" step="any" name="MaxTemp_C" required><br>

            <label>Ambient Temperature (°C)</label><br>
            <input type="number" step="any" name="AmbientTemp_C" required><br>

            <label>Moisture Detected</label><br>
            <select name="MoistureDetected">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br><br>

            <button type="submit">Predict</button>
        </form>

        {% if result %}
            <h3>Result</h3>
            <p class="{{ 'safe' if result.label == 'SAFE' else 'risk' }}">
                {{ result.label }}
            </p>
            <p>Safe Probability: {{ result.safe_prob }}</p>
            <p>Risk Probability: {{ result.risk_prob }}</p>
        {% endif %}
    </body>
    </html>
    """, result=result)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
