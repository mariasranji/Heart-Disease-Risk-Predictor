from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Define the absolute path to your saved artifacts
MODEL_PATH = r"C:\Users\Maria Susan Ranji\Desktop\Nest\kaggle\HEART_DISEASE_RISK\best_heart_disease_model.pkl"

# Load the saved model, scaler, and features once when the app starts
try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    scaler = artifacts['scaler']
    saved_features = artifacts['features']
    print("Model, Scaler, and Features loaded successfully into Flask!")
except Exception as e:
    print(f"Critical Error loading model file: {e}")

@app.route('/')
def home():
    """Renders the HTML form on the home screen for browser users."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles BOTH HTML web form data and raw JSON payloads dynamically."""
    try:
        # Check if the request contains raw JSON data (Postman method)
        if request.is_json:
            data = request.get_json()
            source_is_json = True
        # Otherwise, assume it's coming from the browser HTML form
        else:
            data = request.form
            source_is_json = False

        if not data:
            if source_is_json:
                return jsonify({"status": "error", "message": "No JSON data provided"}), 400
            else:
                return render_template('index.html', prediction_text="Error: No form data received.")

        # Extract values seamlessly from either source and enforce proper types
        input_data = {
            'age': float(data['age']),
            'sex': int(data['sex']),
            'cp': int(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': int(data['fbs']),
            'restecg': int(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': int(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': int(data['slope']),
            'ca': int(data['ca']),
            'thal': int(data['thal'])
        }
        
        # Preprocessing pipeline
        df_new = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_new, columns=['cp', 'restecg', 'slope', 'thal'])
        df_aligned = df_encoded.reindex(columns=saved_features, fill_value=0)
        df_scaled = scaler.transform(df_aligned)
        
        # Generate model inference outputs
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]
        
        status_label = "High Risk of Heart Disease" if prediction == 1 else "Normal / Low Risk"
        probability_pct = f"{probability:.2%}"
        
        # RETURN STRATEGY: Customize response format based on where the request originated
        if source_is_json:
            # Send clean data directly back to Postman JSON console
            return jsonify({
                "status": "success",
                "prediction_class": int(prediction),
                "prediction_label": status_label,
                "risk_probability": probability_pct
            }), 200
        else:
            # Send rendered webpage back to the Web Browser
            return render_template(
                'index.html', 
                prediction_text=status_label, 
                probability_text=probability_pct,
                form_values=request.form
            )
        
    except KeyError as e:
        error_msg = f"Missing required payload field: {str(e)}"
        return jsonify({"status": "error", "message": error_msg}), 400 if request.is_json else render_template('index.html', prediction_text=error_msg)
    except Exception as e:
        error_msg = f"Error in processing pipeline: {str(e)}"
        return jsonify({"status": "error", "message": error_msg}), 500 if request.is_json else render_template('index.html', prediction_text=error_msg)

if __name__ == '__main__':
    app.run(debug=True)