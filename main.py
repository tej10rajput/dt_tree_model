from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import CONFIG

with open(CONFIG.MODEL_PATH,'rb') as file:
    model = pickle.load(file)

with open(CONFIG.ASSET_PATH,'r') as file:
    asset = json.load(file)
col = asset['columns']
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_data", methods = ["POST"])
def data():
    input_data = request.form
    print(input_data)
    
    data = np.zeros(len(col))
    data[0] = input_data['fixed acidity']
    data[1] = input_data['volatile acidity']
    data[2] = input_data['citric acid']
    data[3] = input_data['residual sugar']
    data[3] = input_data['chlorides']
    data[3] = input_data['free sulfur dioxide']
    data[3] = input_data['total sulfur dioxide']
    data[3] = input_data['density']
    data[3] = input_data['pH']
    data[3] = input_data['sulphates']
    data[3] = input_data['alcohol']

    
    result = model.predict([data])
    print(result)

    if result[0] == 3:
        wine_quality = "3"
    if result[0] == 4:
        wine_quality = "4"
    if result[0] == 5:
        wine_quality = "5"
    if result[0] == 6:
        wine_quality = "6"
    if result[0] == 7:
        wine_quality = "7"
    if result[0] == 8:
        wine_quality = "8"

    return render_template("index.html",PREDICT_VALUE=wine_quality)

if __name__ == "__main__":
    app.run(host=CONFIG.HOST_NAME, port= CONFIG.PORT_NUMBER)