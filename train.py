# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 19:47:42 2022

@author: nwenz
"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump



def train():
  #  path =r'C:\Users\nwenz\Desktop\P7_scoring/'
 #   df = pd.read_csv('df_processed.csv')
    
   train_df = pd.read_csv('df_train.csv', nrows =10000)

    
    
    
    
   train_df = train_df.dropna(axis=1).copy()
   train_columns = list(train_df.columns)
    
    

    
   target = train_df.TARGET.copy()
    
   target.dropna(inplace = True)
    

    
   X = train_df.drop(columns = 'TARGET').copy()

    
    
   from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
   from sklearn.svm import SVC
   svm = SVC()
   svm_trained = svm.fit(X,target)
    
   lr = LogisticRegression()
    
   lr_trained = lr.fit(X,target)
    
   from joblib import dump
   dump(svm_trained, 'Inference_svm.joblib')
            
if __name__ == '__main__':
  train()