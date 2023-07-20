#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import m2cgen as m2c 

# In[10]:


data=pd.read_csv("Covid Dataset.csv")


# In[11]:


X=data.iloc[:,0:21]
Y=data.iloc[:,21:22]


# In[12]:

pd.options.mode.chained_assignment = None
x,xtest,y,ytest=train_test_split(X,Y,train_size=0.8)
dic={'Yes':1,'No':0}
for i in x.columns:
    x[i].replace(dic,inplace=True)
for i in xtest.columns:
    xtest[i].replace(dic,inplace=True)
for i in ytest:
    ytest[i].replace(dic,inplace=True)
for i in y:
    y[i].replace(dic,inplace=True)


import xgboost as xgb
xgbClfr=xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss').fit(x,y)
xgbPred=xgbClfr.predict(xtest)

model_to_javascript = m2c.export_to_javascript(xgbClfr)  

print("done")
print(model_to_javascript)
