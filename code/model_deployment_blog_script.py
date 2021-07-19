#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:46:33 2021

@author: siddharthvohra
"""

# Importing Libraries
import pandas as pd
import numpy as np
from flask import Flask,jsonify,request
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import os
from flasgger import Swagger
import flasgger



def train_and_save_model():
    '''This function creates and saves a Binary Logistic Regression
    Classifier in the current working directory
    named as LogisticRegression.pkl
    '''

    ## Creating Dummy Data for Classificaton from sklearn.make_classification
    
    ## n_samples = number of rows/number of samples
    ## n_features = number of total features
    ## n_classes = number of classes - two in case of binary classifier
    X,y = make_classification(n_samples = 1000,n_features = 4,n_classes = 2)
    
    
    ## Train Test Split for evaluation of data - 20% stratified test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)
    
    ## Building Model
    logistic_regression = LogisticRegression(random_state=42)
    
    ## Training the Model
    logistic_regression.fit(X_train,y_train)
    
    ## Getting Predictions
    predictions = logistic_regression.predict(X_test)
    
    ## Analyzing valuation Metrics
    print("Accuracy Score of Model : "+str(accuracy_score(y_test,predictions)))
    
    print("Classification Report : ")
    print(str(classification_report(y_test,predictions)))
    
    ## Saving Model in pickle format
    ## Exports a pickle file named Logisitc Regrssion in current working directory
    output_path = os.getcwd()
    file_name = '/LogisticRegression.pkl'
    output  = open(output_path+file_name,'wb')
    pickle.dump(logistic_regression,output)
    output.close()
    
#train_and_save_model()

app = Flask(__name__)
Swagger(app)

@app.route('/predict',methods = ['GET']) 
def get_predictions():
    
    """
    A simple Test API that returns the predicted class given the  4 parameters named feature 1 ,feature 2 , feature 3 and feature 4
    ---
    
        
    parameters:
       - name: feature_1
         in: query
         type: number
         required: true

       - name: feature_2
         in: query
         type: number
         required: true

       - name: feature_3
         in: query
         type: number
         required: true
        
       - name: feature_4
         in: query
         type: number
         required: true
                          
    responses:
        
        
        200:
            description : predicted Class
    """
        
    ## Getting Features from Swagger UI
    feature_1 = int(request.args.get("feature_1"))
    feature_2 = int(request.args.get("feature_2"))
    feature_3 = int(request.args.get("feature_3"))
    feature_4 = int(request.args.get("feature_4"))
    
#    feature_1 = 1
#    feature_2 = 2
#    feature_3 = 3
#    feature_4 = 4
    
    test_set = np.array([[feature_1,feature_2,feature_3,feature_4]])
    
    ## Loading Model
    infile = open('LogisticRegression.pkl','rb')
    model = pickle.load(infile)
    infile.close()
    
    ## Generating Prediction
    preds = model.predict(test_set)
    
    return jsonify({"class_name":str(preds)})


@app.route('/predict_home/',methods = ['GET']) 
def get_predictions_home():
    
    feature_1 = 1
    feature_2 = 2
    feature_3 = 3
    feature_4 = 4
    
    test_set = np.array([[feature_1,feature_2,feature_3,feature_4]])
    
    ## Loading Model
    infile = open('LogisticRegression.pkl','rb')
    model = pickle.load(infile)
    infile.close()
    
    ## Generating Prediction
    preds = model.predict(test_set)
    
    return jsonify({"class_name":str(preds)})        
    
if __name__=='__main__':
    app.run(debug = True)
    
    ## Visit Base URL  /apidocs/
    

