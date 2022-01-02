# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 19:27:31 2021

@author: lilian
"""


from flask import Flask, render_template, url_for, request, jsonify
#from flask_restful import API, Resource
from model_incident import RedditClassificationModel





model = RedditClassificationModel('models/models_trained/nnlm-en-dim128.json', 'models/models_trained/nnlm-en-dim128_model.h5')





app = Flask(__name__)

@app.route('/')
def interface():
    return render_template('interface.html')

@app.route('/predict', methods = ['POST'])
def predict():
    message = request.form['message']
    data = [message]
    pred = model.predict_emotion(data)
    return render_template('result.html', pred= pred)

if __name__=='__main__':
   
    
  
    app.run(debug=True)
  
