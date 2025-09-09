from flask import Flask, request, jsonify
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Determine absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_ai.pkl")

# Load trained model
try:
    with open(MODEL_PATH, "rb") as f:
        ai = pickle.load(f)
    logging.info(f"Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Upload 'ai_model.pkl'.")
    ai = None
except Exception as e:
    logging.error(f"Error loading model: {e}")
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
        ir = float(ir)  # support int or float
    except ValueError:
        return jsonify({"error": "'ir' must be a number"}), 400

    data = [[ir]]
    try:
        prediction = ai.predict(data)[0]
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # dynamic port for hosting
    app.run(host="0.0.0.0", port=port)
