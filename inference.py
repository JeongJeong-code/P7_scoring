# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 19:59:15 2022

@author: nwenz
"""
import os
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression




def inference():   
#    path =r'C:\Users\nwenz\Desktop\P7_scoring
    test_df = pd.read_csv('df_test.csv')
    train_df = pd.read_csv('df_train.csv',nrows = 10000)

    
    
    
    
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
    inference()