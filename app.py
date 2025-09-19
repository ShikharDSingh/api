from flask import Flask, request, jsonify
from flask_cors import CORS  
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # ✅ allow cross-origin requests (important for frontend apps)

def get_cleaned_data(json_data):
    gestation = float(json_data['gestation'])
    parity = int(json_data['parity'])
    age = float(json_data['age'])
    height = float(json_data['height'])
    weight = float(json_data['weight'])
    smoke = float(json_data['smoke'])

    cleaned_data = {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }
    return cleaned_data

# ✅ health check route for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Baby Prediction API. Use POST /predict with JSON data."})

@app.route("/predict", methods=['POST'])
def get_prediction():
    try:
        baby_data_json = request.get_json(force=True)  # ✅ read JSON body
        baby_data_cleaned = get_cleaned_data(baby_data_json)

        baby_df = pd.DataFrame(baby_data_cleaned)

        with open("model.pkl", 'rb') as obj:
            model = pickle.load(obj)

        prediction = model.predict(baby_df)
        prediction = round(float(prediction), 2)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Render sets PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)
