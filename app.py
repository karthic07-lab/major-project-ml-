import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

# --------------------------------
# Flask App
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Load Model & Scaler
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model-2.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "standard_scaler.joblib"))

# --------------------------------
# FULL FEATURE LIST (TRAINING ORDER)
# --------------------------------
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

# --------------------------------
# Home (Health Check)
# --------------------------------
@app.route("/")
def home():
    return "EV Battery Thermal Runaway Prediction API is running."

# --------------------------------
# API ENDPOINT (JSON – IoT / Postman)
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build full feature vector
        full_data = {}
        for col in FEATURE_COLUMNS:
            full_data[col] = data.get(col, 0)

        df = pd.DataFrame(
            [[full_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)

        return jsonify({
            "prediction": int(pred[0]),
            "safe_probability": float(prob[0][0]),
            "risk_probability": float(prob[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------
# TRENDY BROWSER UI
# --------------------------------
@app.route("/ui", methods=["GET", "POST"])
def ui():
    result = None
    error = None

    if request.method == "POST":
        try:
            # Read UI inputs safely
            ui_data = {
                "PackVoltage_V": float(request.form.get("PackVoltage_V", 0)),
                "ChargeCurrent_A": float(request.form.get("ChargeCurrent_A", 0)),
                "SOC_%": float(request.form.get("SOC_%", 0)),
                "MaxTemp_C": float(request.form.get("MaxTemp_C", 0)),
                "AmbientTemp_C": float(request.form.get("AmbientTemp_C", 0)),
                "MoistureDetected": int(request.form.get("MoistureDetected", 0))
            }

            # Fill remaining features with 0
            full_data = {}
            for col in FEATURE_COLUMNS:
                full_data[col] = ui_data.get(col, 0)

            df = pd.DataFrame(
                [[full_data[col] for col in FEATURE_COLUMNS]],
                columns=FEATURE_COLUMNS
            )

            df_scaled = scaler.transform(df)
            pred = model.predict(df_scaled)
            prob = model.predict_proba(df_scaled)

            result = {
                "status": "SAFE" if pred[0] == 0 else "THERMAL RUNAWAY RISK",
                "safe": round(prob[0][0] * 100, 2),
                "risk": round(prob[0][1] * 100, 2)
            }

        except Exception as e:
            error = str(e)

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>EV Thermal Predictor</title>
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    font-family: Arial;
    color: white;
}
.card {
    width: 420px;
    margin: 60px auto;
    padding: 30px;
    background: rgba(255,255,255,0.1);
    border-radius: 16px;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
}
input, select, button {
    width: 100%;
    padding: 10px;
    margin-bottom: 12px;
    border-radius: 8px;
    border: none;
}
button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    font-size: 16px;
    cursor: pointer;
}
.safe { color: #00ff88; font-weight: bold; }
.risk { color: #ff4d4d; font-weight: bold; }
.error { color: orange; }
</style>
</head>

<body>
<div class="card">
<h2>EV Battery Thermal Runaway Predictor</h2>

<form method="post">
<input name="PackVoltage_V" placeholder="Pack Voltage (V)" required>
<input name="ChargeCurrent_A" placeholder="Charge Current (A)" required>
<input name="SOC_%" placeholder="State of Charge (%)" required>
<input name="MaxTemp_C" placeholder="Max Temperature (°C)" required>
<input name="AmbientTemp_C" placeholder="Ambient Temperature (°C)" required>

<select name="MoistureDetected">
    <option value="0">Moisture: No</option>
    <option value="1">Moisture: Yes</option>
</select>

<button type="submit">Predict</button>
</form>

{% if result %}
    <h3 class="{{ 'safe' if result.status == 'SAFE' else 'risk' }}">
        {{ result.status }}
    </h3>
    <p>Safe Probability: {{ result.safe }}%</p>
    <p>Risk Probability: {{ result.risk }}%</p>
{% endif %}

{% if error %}
    <p class="error">Error: {{ error }}</p>
{% endif %}
</div>
</body>
</html>
""", result=result, error=error)

# --------------------------------
# Run Server (Render-safe)
# --------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
