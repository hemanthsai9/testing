from flask import Flask, request, jsonify
import pickle

# Load trained model (trained locally and saved as trained_ai.pkl)
with open("trained_ai.pkl", "rb") as f:
    ai = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def homepage():
    return "✅ AI Server Running"

@app.route("/predict")
def predict():
    ir = request.args.get("ir")
    if ir is None:
        return jsonify({"error": "Missing query parameter 'ir'"}), 400
    
    try:
        ir = int(ir)
    except ValueError:
        return jsonify({"error": "'ir' must be an integer"}), 400

    data = [[ir]]
    result = ai.predict(data)[0]  # prediction
    return jsonify({"prediction": str(result)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000)
