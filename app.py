from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"./models/final_stacking_model1.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        chest_pain = int(request.form["chest_pain"])
        resting_bp = int(request.form["resting_bp"])
        cholesterol = int(request.form["cholesterol"])
        fasting_bs = int(request.form["fasting_bs"])
        rest_ecg = int(request.form["rest_ecg"])
        max_heart_rate = int(request.form["max_heart_rate"])
        exercise_angina = int(request.form["exercise_angina"])
        st_depression = float(request.form["st_depression"])
        st_slope = int(request.form["st_slope"])

        # Prepare input array for prediction
        input_data = np.array([[age, gender, chest_pain, resting_bp, cholesterol, 
                                fasting_bs, rest_ecg, max_heart_rate, exercise_angina, 
                                st_depression, st_slope]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template("index.html", prediction=result, 
                               message="Consult a doctor if necessary!" if prediction == 1 else "You are healthy!")

    except Exception as e:
        return render_template("index.html", prediction="Error", message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
