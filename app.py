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

FEATURE_COLUMNS = [
    "PackVoltage_V","CellVoltage_V","DemandVoltage_V","ChargeCurrent_A",
    "DemandCurrent_A","SOC_%","MaxTemp_C","MinTemp_C","AvgTemp_C",
    "AmbientTemp_C","InternalResistance_mOhm","StateOfHealth_%",
    "VibrationLevel_mg","MoistureDetected","ChargePower_kW","Pressure_kPa",
    "ChargingStage_Handshake","ChargingStage_Parameter_Config",
    "ChargingStage_Recharge","BMS_Status_OK","BMS_Status_Warning"
]

@app.route("/")
def home():
    return "EV Battery Thermal Runaway Prediction API is running."

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
# TRENDY UI
# -----------------------------
@app.route("/ui", methods=["GET", "POST"])
def ui():
    result = None

    if request.method == "POST":
        data = {
            "PackVoltage_V": float(request.form["PackVoltage_V"]),
            "ChargeCurrent_A": float(request.form["ChargeCurrent_A"]),
            "SOC_%": float(request.form["SOC_%"]),
            "MaxTemp_C": float(request.form["MaxTemp_C"]),
            "AmbientTemp_C": float(request.form["AmbientTemp_C"]),
            "MoistureDetected": int(request.form["MoistureDetected"])
        }

        for col in FEATURE_COLUMNS:
            data.setdefault(col, 0)

        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)

        result = {
            "status": "SAFE" if pred[0] == 0 else "THERMAL RUNAWAY RISK",
            "safe": round(prob[0][0]*100, 1),
            "risk": round(prob[0][1]*100, 1)
        }

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>EV Thermal Predictor</title>
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
    color: white;
}
.card {
    max-width: 420px;
    margin: 60px auto;
    padding: 30px;
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}
h2 {
    text-align: center;
    margin-bottom: 25px;
}
label {
    font-size: 14px;
}
input, select {
    width: 100%;
    padding: 10px;
    margin: 8px 0 16px;
    border-radius: 8px;
    border: none;
    outline: none;
}
button {
    width: 100%;
    padding: 12px;
    background: #00c6ff;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    cursor: pointer;
}
button:hover {
    opacity: 0.9;
}
.result {
    margin-top: 25px;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
.safe {
    background: rgba(0,255,100,0.2);
    color: #00ff88;
}
.risk {
    background: rgba(255,0,0,0.25);
    color: #ff4d4d;
}
.badge {
    font-size: 22px;
    font-weight: bold;
}
</style>
</head>

<body>
<div class="card">
<h2>EV Battery Thermal Predictor</h2>

<form method="post">
<label>Pack Voltage (V)</label>
<input type="number" step="any" name="PackVoltage_V" required>

<label>Charge Current (A)</label>
<input type="number" step="any" name="ChargeCurrent_A" required>

<label>State of Charge (%)</label>
<input type="number" step="any" name="SOC_%" required>

<label>Max Temperature (°C)</label>
<input type="number" step="any" name="MaxTemp_C" required>

<label>Ambient Temperature (°C)</label>
<input type="number" step="any" name="AmbientTemp_C" required>

<label>Moisture Detected</label>
<select name="MoistureDetected">
  <option value="0">No</option>
  <option value="1">Yes</option>
</select>

<button type="submit">Predict Risk</button>
</form>

{% if result %}
<div class="result {{ 'safe' if result.status == 'SAFE' else 'risk' }}">
    <div class="badge">{{ result.status }}</div>
    <p>Safe Probability: {{ result.safe }}%</p>
    <p>Risk Probability: {{ result.risk }}%</p>
</div>
{% endif %}
</div>
</body>
</html>
""", result=result)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
