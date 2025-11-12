from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained Decision Tree model
model = joblib.load("captcha_model.pkl")

def extract_features(cursor_data):
    if len(cursor_data) < 2:
        return None  # Not enough data to process

    timestamps = np.array([entry["clientTimestamp"] for entry in cursor_data])
    x_positions = np.array([entry["x"] for entry in cursor_data])
    y_positions = np.array([entry["y"] for entry in cursor_data])

    # Compute time differences
    timestamp_diff = np.diff(timestamps)
    timestamp_diff[timestamp_diff == 0] = np.nan  # Avoid division by zero

    # Compute distances
    distances = np.sqrt(np.diff(x_positions) ** 2 + np.diff(y_positions) ** 2)

    # Compute speed and acceleration
    speeds = distances / timestamp_diff
    accelerations = np.diff(speeds) / timestamp_diff[1:]

    # Extract required features
    avgSpeed = np.nanmean(speeds)  # Mean speed
    accelVar = np.nanvar(accelerations)  # Acceleration variance
    jitter = np.nanstd(distances)  # Standard deviation of distances
    smoothness = 1 / (1 + jitter)  # Smoothness score (lower jitter = higher smoothness)
    dwellTime = np.mean(timestamp_diff)  # Mean dwell time
    midCrossings = ((x_positions[:-1] < np.median(x_positions)) & (x_positions[1:] >= np.median(x_positions))).sum()

    return [avgSpeed, accelVar, jitter, smoothness, dwellTime, midCrossings]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    cursor_data = data.get("cursorData", [])

    # Extract features
    features = extract_features(cursor_data)

    if features is None:
        return jsonify({"error": "Not enough data to process"}), 400

    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # Predict using Decision Tree model
    prediction = model.predict(features)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
