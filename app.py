from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
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






if __name__ == '__main__':
    app.run(debug = True)