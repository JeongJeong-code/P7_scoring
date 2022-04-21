# -*- coding: utf-8 -*-

import pandas as pd
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble  import RandomForestClassifier
import lime.lime_tabular
import json
#class Id(BaseModel):
#    idx: float

#loaded_model = load('rfc2.joblib')
#path = r'C:\Users\nwenz\Desktop\heroku_test\app/'
path = r'/app/'
X_pred = pd.read_csv(path+'X_test_export.csv')


X_train = pd.read_csv(path+'X_train_export.csv');
y_train = pd.read_csv(path+'y_train_export.csv')
top_feat = pd.read_csv(path+'top_feat.csv')
feat = top_feat.feat_names
X_train =X_train[list(feat)]
loaded_model = RandomForestClassifier( max_depth=5, n_estimators=5, min_samples_split=10, min_samples_leaf=10)

loaded_model.fit(X_train,y_train['TARGET'])



def feat_importance(features,cols):
  feat_indexes =np.flatnonzero(features)
  feat_values = features[feat_indexes]
  feat_names = cols[feat_indexes]

  feat_indexes = pd.Series(feat_indexes,name ='feat_indexes')#.reset_index(drop=True)
  feat_values = pd.Series(feat_values,name ='feat_values')#.reset_index(drop=True)
  feat_names = pd.Series(feat_names,name = 'feat_names')#.reset_index(drop=True)

  features_df = pd.concat([feat_names,feat_indexes,feat_values],axis =1)
  features_df.sort_values(by=['feat_values'],ascending =False,inplace =True)
  return features_df








app = FastAPI()

class Customer(BaseModel):
    UserID: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/feature")
def get_feat():
       model_feat = feat_importance(loaded_model.feature_importances_,X_train.columns)
       feat_json = model_feat.to_json()
       return feat_json
   
@app.post('/explain')
def get_explain(idx:Customer):
   index = idx.UserID
   
   X_sample = X_pred.loc[index,feat]
   #X_sample = X_sample[top_feat]
   
   X_arr = X_sample.values.reshape(1,-1) 
   line_pred = loaded_model.predict(X_arr)   
   X_lime = np.array(X_train)
   explainer =lime.lime_tabular.LimeTabularExplainer(X_lime, mode='classification', feature_names=feat)
   X_arr_lime = np.reshape(X_arr,[49,])
   exp = explainer.explain_instance(X_arr_lime, loaded_model.predict_proba, num_features=10, top_labels=1)
   exp_list = exp.as_list(label = float(line_pred))
   json_exp = json.dumps(dict(exp_list))
   return json_exp

@app.post("/prediction")
def read_root(idx : Customer):
   index = idx.UserID
   X_sample = X_pred.loc[index,feat]
   X_arr = X_sample.values.reshape(1,-1)
   print(np.shape(np.array(X_arr)))
   pred_name = loaded_model.predict(X_arr)#[0]
   pred_proba = loaded_model.predict_proba(X_arr)
   prob_0 = pred_proba[0,0]
   prob_1 = pred_proba[0,1]
   
   

   

   #print('prob_1=',prob_1)
   #print(np.shape(np.array(pred_proba)))
#   print(prob_1)
   #pred_dict.update({'pred_value':id})
   #id2 = idx.UserID +1
   id2 = int(pred_name[0])
   
   result = {"predi" :id2,
             "prob_0": prob_0,
             "prob_1":prob_1
                 }
   return result






