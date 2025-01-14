from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained Linear Regression model
model = joblib.load('linear_model.pkl')

app = Flask('stock_predict')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON for features
        data = request.get_json()
        features =  pd.Series(data).to_numpy().reshape(1, -1)

        # Make prediction
        prediction = float(model.predict(features)[0])

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 9990)
