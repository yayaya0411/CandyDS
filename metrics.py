
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, log_loss
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# https://scikit-learn.org/stable/modules/model_evaluation.html

def clf_metrics(y, y_pred):
    """
    caculate classification machines performance

    Args:
    ----------
        y : y actual
        y_pred :  y predict
    Return:
    ----------
        d : Train varable
    """
    d={}
    d["accuracy"] = round(accuracy_score(y, y_pred),4)
    d["precision"] = round(precision_score(y, y_pred),4)
    d["recall"] = round(recall_score(y, y_pred),4)
    d["f1"] = round(f1_score(y, y_pred),4)
    d["AUC"] = round(roc_auc_score(y, y_pred),4)
    d["log_loss"] = round(log_loss(y, y_pred),4)
    return d

def reg_metrics(y, y_pred):
    """
    caculate regression machines performance

    Args:
    ----------
        y : y actual
        y_pred :  y predict
    Return:
    ----------
        d : Train varable
    """
    d={}
    d["MSE"] = round(mean_squared_error(y,y_pred),4)
    d["RMSE"] = round(np.sqrt(mean_squared_error(y, y_pred)),4)
    d["R2"] = round(r2_score(y,y_pred),4)
    return d

def clf_report(y, y_pred):
    """
    caculate classification machines report

    Args:
    ----------
        y : y actual
        y_pred :  y predict
    Return:
    ----------
        confusionmatrix : confusion matrix
        report : some metrics report
    """
    confusionmatrix = confusion_matrix(y,y_pred)
    report = classification_report(y, y_pred)
    print("Confusion Matrix")
    print(confusionmatrix)
    print()
    print("Classification Report")
    print(report)
    #return confusionmatrix ,report
