
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import auc,  confusion_matrix
import numpy as np
# metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html

def clf_metrics(y, y_pred):
    d={}
    d["accuracy"] = round(accuracy_score(y, y_pred),4)
    d["precision"] = round(precision_score(y, y_pred),4)
    d["recall"] = round(recall_score(y, y_pred),4)
    d["f1"] = round(f1_score(y, y_pred),4)
    return d

def reg_metrics(y, y_pred):
    d={}
    d["MSE"] = round(mean_squared_error(y,y_pred),4)
    d["RMSE"] = round(np.sqrt(mean_squared_error(y, y_pred)),4)
    d["R2"] = round(r2_score(y,y_pred),4)
    return d
