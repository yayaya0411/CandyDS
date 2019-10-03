import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, Normalizer


def check_missing(df,threshold=0.3,list=True):
    """
    impute missing values

    Args:
    ----------
        df : dataframe
        threshold : missing ratio to contral drop column
        list : if list == True, return drop_list
    Return:
    ----------
        drop_list : list
    """
    df_list = df.columns
    drop_list = []
    total_cell = np.product(df.shape)
    total_missing_cell = df.isnull().sum().sum()
    total_missing_ratio = round(total_missing_cell/total_cell,2)
    print("Missing ratio : " )
    print(" Total Missing ratio : " +str(total_missing_ratio))
    for var in df_list:
        missing_ratio = round(1-(df[var].count() / len(df[var])),2)
        if missing_ratio > threshold:
            drop_list.append(var)
        print("  "+var +" : "+str(missing_ratio))
    if len(drop_list) > 0:
        df = df.drop(columns=drop_list)
        print("\ndrop missing ratio > "+str(threshold) +" threshold columns : ")
        print(*drop_list,sep=', ')
    print()
    if list:
        return drop_list


def fill_up(df,strategy=None,drop = True,threshold = 0.3):
    """
    impute missing values

    Args:
    ----------
        df : dataframe
        threshold : missing ratio to contral drop column
        strategy : which value to replace nan value
        drop : if drop == True, drop drop_list columns
    Return:
    ----------
        df : dataframe
    """
    df_list = df.columns
    if drop:
        drop_list = check_missing(df,threshold = threshold)
    else :
        drop_list = []
    df = df.drop(columns = drop_list)
    if strategy == None or "median":
        for i in df.select_dtypes(include=["float64", "int"]).columns:
            df[i] = df[i].fillna(df[i].median())
            print(i+" imputer with median which is "+str(df[i].median()))
        for i in df.select_dtypes(include=["object"]).columns:
            df[i] = df[i].fillna(df[i].mode().values[0])
            print(i+" imputer with mode which is "+str(df[i].mode().values[0]))
    elif strategy == "mean":
        for i in df.select_dtypes(include=["float64", "int"]).columns:
            df[i] = df[i].fillna(df[i].mean())
            print(i+" imputer with mean which is "+str(df[i].mean()))
        for i in df.select_dtypes(include=["object"]).columns:
            df[i] = df[i].fillna(df[i].mode().values[0])
            print(i+" imputer with mode which is "+str(df[i].mode().values[0]))
    return df

def check_objvalues(df,distinct_values=10,drops = False):
    """
    transform object into dummy

    Args:
    ----------
        df : dataframe
        distinct_values : check obj feature distinct values
        drops : return drop_list or not
    Return:
    ----------
        df_new : dataframe
    """
    obj_list = list(df.select_dtypes("object"))
    print()
    print(*obj_list,sep=", ")
    print("  feature may transform to dummy")
    drop_list = []
    for i in obj_list:
        counter = len(df[i].unique().tolist())
        print(i+" have "+str(counter)+" distinct values")
        if counter > distinct_values:
            drop_list.append(i)
            print("  "+i + " feature have more than "+ str(distinct_values) +" kinds of valus")
    print()
    if drops == True:
        print(*drop_list,sep=", ")
        print("columns which have too many value, return drop_list to drop columns")
        return drop_list
    elif drops == False:
        print(*drop_list,sep=", ")
        print("  columns may have too many values, if want drop columns, set drops = True to get drop_list")

def get_dummy(df,drop_list=[]):
    """
    transform object into dummy

    Args:
    ----------
        df : dataframe
        drop_list : drop columns that you don't want to transform
    Return:
    ----------
        df_new : dataframe
    """
    df = df.drop(columns=drop_list)
    obj_list = list(df.select_dtypes("object"))
    dummy = pd.get_dummies(df[obj_list])
    df_new = df.drop(obj_list, axis=1)
    df_new = pd.concat([df_new, dummy], axis=1)
    return df_new

def get_scaler(df, scaler='M'):
    """
    to get scaler object

    Args:
    ----------
        df : dataframe
        scaler : method to scaler
                S means StandardScaler, base on column-wise ,changing values to (0,1) standard deviation
                M means MinMaxScaler, rescales the data set that all feature values are in the range [0, 1]
                                    , is common used for measures how far apart data point like SVM, KNN etc.
                N means Normalizer, base on row-wise, rescaled independently of other samples so that its norm (l1 or l2) equals one
                                    , is a common operation for text classification or clustering for instance
                see more : https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    Return:
    ----------
        scaler : scaler object
        scaler_list : columns to scaler
    """
    scaler_list = list(df.select_dtypes(include=["float64", "int64"]))
    scaler_df = df[scaler_list]
    print()
    print(*scaler_list,sep=", ")
    print("  Ready to scale columns")
    if scaler == 'S':
        scaler = StandardScaler().fit(scaler_df)
        print("Use StandardScaler")
    if scaler == 'M':
        scaler = MinMaxScaler().fit(scaler_df)
        print("Use MinMaxScaler")
    if scaler == 'N':
        scaler = Normalizer().fit(scaler_df)
        print("Use Normalizer")
    return scaler, scaler_list

def scale(df, scaler, scaler_list):
    """
    to get scaler dataframe

    Args:
    ----------
        df : dataframe
        scaler : method to scaler
        scaler_list : columns to scaler

    Return:
    ----------
        scaler_df : data_frame
    """
    tmp_df = df.drop(scaler_list, axis=1)
    scale_df = scaler.transform(df[scaler_list])
    scale_df = pd.DataFrame(scale_df, columns=scaler_list)
    scale_df = pd.concat([tmp_df, scale_df], axis=1)
    return scale_df
