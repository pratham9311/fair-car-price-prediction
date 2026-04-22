"""
FairCar – Flask Backend API
============================
Connects the FairCar frontend to the trained Random Forest model.

Usage:
  1. Train and save the model first (run train_model.py)
  2. Run:  python app.py
  3. API is served at http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "cars24_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Please run train_model.py first to train and save the model."
    )

model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Serve the FairCar frontend — visit http://localhost:5000 in your browser."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

# ── Health check (API only) ───────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "FairCar Price Prediction API"})


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "brand":     "Hyundai",
        "model":     "Creta",
        "year":      2019,
        "km_driven": 35000,
        "fuel_type": "Petrol",
        "owner":     "First",
        "drive":     "FWD",
        "car_type":  "SUV"
    }

    Returns:
    {
        "predicted_price": 750000.0,
        "formatted_price": "₹7,50,000",
        "currency": "INR"
    }
    """
    data = request.get_json(force=True)

    # ── Validate required fields ──────────────────────────────────────────
    required = ["brand", "model", "year", "km_driven", "fuel_type", "owner", "drive", "car_type"]
    missing = [f for f in required if f not in data or data[f] in ("", None)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # ── Type coercion ─────────────────────────────────────────────────────
    try:
        year     = int(data["year"])
        km       = int(data["km_driven"])
    except (ValueError, TypeError):
        return jsonify({"error": "year and km_driven must be numeric"}), 400

    # ── Build input DataFrame (matches training schema) ───────────────────
    input_df = pd.DataFrame([{
        "brand":     str(data["brand"]).strip(),
        "model":     str(data["model"]).strip(),
        "year":      year,
        "km_driven": km,
        "fuel_type": str(data["fuel_type"]).strip(),
        "owner":     str(data["owner"]).strip(),
        "drive":     str(data["drive"]).strip(),
        "car_type":  str(data["car_type"]).strip(),
    }])

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        raw_price = model.predict(input_df)[0]
        price     = max(float(raw_price), 0.0)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ── Format Indian numbering (e.g. ₹7,50,000) ─────────────────────────
    price_int = int(round(price, -3))   # round to nearest ₹1,000
    formatted = "₹{:,}".format(price_int)
    # Convert western commas to Indian format
    formatted = _to_indian_format(price_int)

    return jsonify({
        "predicted_price": price_int,
        "formatted_price": formatted,
        "currency": "INR"
    })


def _to_indian_format(n: int) -> str:
    """Format an integer in Indian numbering system (₹X,XX,XX,XXX)."""
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    # Last 3 digits
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return "₹" + result.lstrip(",")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
