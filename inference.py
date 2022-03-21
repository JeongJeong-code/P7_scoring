# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 19:59:15 2022

@author: nwenz
"""
import os
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from flask import Flask,request

port = int(os.environ.get('PORT', 5000))
#app = Flask(__name__)



#◙@app.route('/inference')
def inference():   
#    path =r'C:\Users\nwenz\Desktop\P7_scoring
    test_df = pd.read_csv('df_test.csv',nrows=10)
    train_df = pd.read_csv('df_train.csv',nrows = 1000)

    
    
    
    
    train_df = train_df.dropna(axis=1).copy()
    train_columns = list(train_df.columns)
    
    
    test_df = test_df[test_df.columns.intersection(train_columns)]
    
    target = train_df.TARGET.copy()
    
    target.dropna(inplace = True)
    
    test_df.drop(columns = 'TARGET',inplace = True)
    test_df.dropna(inplace = True)   
    
    X_test = test_df.copy()

    

    
    # Models training
    
    # Run model
    lr_trained = load('Inference_svm.joblib')
    print(lr_trained.predict(X_test))
        

if __name__ == '__main__':
    #app.run(debug=True,host='0.0.0.0',port=port)
    inference()