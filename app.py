from flask import Flask, request, jsonify
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained model
try:
    with open("trained_ai.pkl", "rb") as f:
        ai = pickle.load(f)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("trained_ai.pkl not found. Make sure the file is uploaded.")
    ai = None

app = Flask(__name__)

@app.route("/")
def homepage():
    return "✅ AI Server Running"

@app.route("/predict")
def predict():
    if ai is None:
        return jsonify({"error": "Model not loaded"}), 500

    ir = request.args.get("ir")
    if ir is None:
        return jsonify({"error": "Missing query parameter 'ir'"}), 400

    try:
        ir = float(ir)  # support integer or float
    except ValueError:
        return jsonify({"error": "'ir' must be a number"}), 400

    data = [[ir]]
    prediction = ai.predict(data)[0]
    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use host-provided PORT
    app.run(host="0.0.0.0", port=port)
