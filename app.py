import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, flash


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
app.config['SECRET_KEY'] = "secret lol"

@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/submit", methods=['POST'])
def submit():

    # The input data
    input_data = [request.form['title']+' '+ request.form['location']+' '+request.form['department']+' '+request.form['profile']+' '+request.form['req']+' '+request.form['ben']+' '+request.form['emptype']+' '+request.form['exp']+' '+request.form['edu']+' '+request.form['indu']+' '+request.form['func']+' '+request.form['des']]


    # convert text to feature vectors
    input_data_features = vectorizer.transform(input_data)

    # making prediction
    prediction = model.predict(input_data_features)
    print(prediction)

    if (prediction[0] == 1):
        flash("FRAUDULENT JOB")
        return render_template('index.html')

    else:
        flash("REAL JOB")
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')