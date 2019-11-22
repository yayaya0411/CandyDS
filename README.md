## Some handy tools for my regular work
----------------------------------------------------------------
examples

```py
# import packages
import numpy as np
import pandas as pd
import os
import pickle

from pretools import sitetool
from pretools import process
from pretools import metrics

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_auc_score,roc_curve,classification_report

#%matplotlib inline
#plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
print("import package done")

# load titanic data
titanic = pd.read_csv('https://raw.githubusercontent.com/yayaya0411/pretools/master/data/titanic/train.csv')
drop = ["PassengerId","Ticket","Name"]
X,y = sitetool.arrange_df(titanic,y="Survived",drops=drop)
process.check_missing(X,threshold = 0.3)

# fill up missing
X = process.fill_up(X,drop=False,strategy = "median",threshold = 0.3)

# get dummy
drop_list = process.check_objvalues(X,drops = True)
X = process.get_dummy(X,drop_list=drop_list)

# scale data
scaler,scaler_list = process.get_scaler(X)
X = process.scale(X,scaler,scaler_list)

# split data
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state = 666)

# built XGB model
model_xgb = xgb.XGBClassifier(objective="binary:logistic", random_state=666)
model_xgb.fit(X, y)
y_pred_XGB = model_xgb.predict(X)

# built rf model
model_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=666)
model_rf.fit(X,y)
model_rf.feature_importances_
y_pred_rf = model_rf.predict(X)

# class & number need use different metrics method
metrics.clf_metrics(y,y_pred_rf)
metrics.reg_metrics(y,y_pred_rf)
metrics.clf_report(y,y_pred_rf)
print(classification_report(y,y_pred_rf))

# save model
model_name = "model_xgb"
model_folder = 'model'
model_path = os.path.join(model_folder,model_name)
pickle.dump(model_xgb,open(model_path,"wb"))

# load model
load_model = pickle.load(open(model_path,"rb"))
load_model.predict(y_train)

```
