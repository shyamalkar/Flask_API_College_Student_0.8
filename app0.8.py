from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("student_placement_0.7.joblib")

# Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API is running",
        "message": "Student Placement Prediction API"
    })

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Arrange features EXACTLY as training order
        features = np.array([[
            data["Age"],
            data["Gender"],
            data["Stream"],
            data["Internships"],
            data["CGPA"],
            data["Hostel"],
            data["HistoryOfBacklogs"]
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)

