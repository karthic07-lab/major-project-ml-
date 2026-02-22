@app.route("/ui", methods=["GET", "POST"])
def ui():
    result = None
    error = None

    if request.method == "POST":
        try:
            # Read only UI inputs safely
            ui_data = {
                "PackVoltage_V": float(request.form.get("PackVoltage_V", 0)),
                "ChargeCurrent_A": float(request.form.get("ChargeCurrent_A", 0)),
                "SOC_%": float(request.form.get("SOC_%", 0)),
                "MaxTemp_C": float(request.form.get("MaxTemp_C", 0)),
                "AmbientTemp_C": float(request.form.get("AmbientTemp_C", 0)),
                "MoistureDetected": int(request.form.get("MoistureDetected", 0))
            }

            # Build full feature vector (21 features)
            full_data = {}
            for col in FEATURE_COLUMNS:
                full_data[col] = ui_data.get(col, 0)

            # Create DataFrame in EXACT order
            df = pd.DataFrame(
                [[full_data[col] for col in FEATURE_COLUMNS]],
                columns=FEATURE_COLUMNS
            )

            # Scale & predict
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
                background: #0f2027;
                font-family: Arial;
                color: white;
            }
            .card {
                width: 420px;
                margin: 60px auto;
                padding: 30px;
                background: rgba(255,255,255,0.08);
                border-radius: 16px;
            }
            input, select, button {
                width: 100%;
                padding: 10px;
                margin-bottom: 12px;
            }
            button {
                background: #00c6ff;
                border: none;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            .safe { color: #00ff88; }
            .risk { color: #ff4d4d; }
            .error { color: orange; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>EV Battery Thermal Predictor</h2>

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
