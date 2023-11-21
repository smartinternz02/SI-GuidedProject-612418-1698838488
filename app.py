import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Use binary mode for opening the model file
with open("C:/Users/Siri/Anaconda_files/TrafficTelligence - Advanced Traffic Volume Estimation with Machine Learning/Flask/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Use binary mode for opening the scaler file
with open("C:/Users/Siri/Anaconda_files/TrafficTelligence - Advanced Traffic Volume Estimation with Machine Learning/Flask/scaler.pkl", "rb") as scale_file:
    scale = pickle.load(scale_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'Day', 'Month', 'Year', 'Hours', 'Minutes', 'Seconds']
    data = pd.DataFrame(features_values, columns=names)
    data = scale.transform(data)
    data = pd.DataFrame(data, columns=names)
    
    prediction = model.predict(data)
    print(prediction)
    text = "Estimated Traffic volume is: " + str(prediction[0])
    
    return render_template("index.html", prediction_text=text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
