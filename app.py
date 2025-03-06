from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    dataframe = pd.DataFrame(data, index = [0])
    data_scaled = scaler.transform(dataframe)
    prediction = model.predict(data_scaled)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict_user', methods = ['POST'])
def user_predict():
    data = [float(x) for x in request.form.values()]
    columns_name = list(request.form.keys())

    dataframe = pd.DataFrame([data], columns = columns_name)
    final_input = scaler.transform(dataframe)
    prediction = model.predict(final_input)[0]
    # prediction_text doit être identique dans html
    return render_template('home.html', prediction_text = f'The predicted price is: {prediction}')


if __name__ == '__main__':
    app.run(debug = True)