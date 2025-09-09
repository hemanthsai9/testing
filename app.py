from flask import Flask, request, jsonify
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained model (make sure trained_ai.pkl is uploaded to server)
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

@app.route("/predict", methods=["POST"])
def predict():
    if ai is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Expect JSON input: {"features": [1,2,3]}
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing JSON body or 'features' key"}), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400

    try:
        # Ensure all elements are numbers
        features = [[float(x) for x in features]]
    except ValueError:
        return jsonify({"error": "All features must be numeric"}), 400

    prediction = ai.predict(features)[0]
    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use host-provided PORT or fallback
    app.run(host="0.0.0.0", port=port)
