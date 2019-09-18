import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, Normalizer

def fill_up(df,threshold=0.3,strategy=None):
    """
    impute missing values

    Args:
    ----------
        df : dataframe
        threshold : missing ratio to contral drop column
        strategy : which value to replace nan value
    Return:
    ----------
        df : dataframe
    """
    df_list = df.columns
    drop_list = []
    print("missing ratio : ")
    for var in df_list:
        missing_ratio = round(1-(df[var].count() / len(df[var])),2)
        if missing_ratio > threshold:
            drop_list.append(var)
        print(var +" : "+str(missing_ratio))
    if len(drop_list) > 0:
        df = df.drop(columns=drop_list)
        print("\ndrop missing ratio > "+str(threshold) +" columns : ")
        print(*drop_list,sep=', ')
        print()
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


def get_dummy(df):
    """
    transform object into dummy

    Args:
    ----------
        df : dataframe
    Return:
    ----------
        df_new : dataframe
    """
    obj_list = list(df.select_dtypes("object"))
    dummy = pd.get_dummies(df[obj_list])
    df_new = df.drop(obj_list, axis=1)
    df_new = pd.concat([df_new, dummy], axis=1)
    return df_new

def get_scaler(df, scaler='S'):
    """
    to get scaler object

    Args:
    ----------
        df : dataframe
        scaler : method to scaler

    Return:
    ----------
        scaler : scaler object
        scaler_list : columns to scaler
    """
    scaler_list = list(df.select_dtypes(include=["float64", "int64"]))
    scaler_df = df[scaler_list]
    if scaler == 'S':
        scaler = StandardScaler().fit(scaler_df)
    if scaler == 'M':
        scaler = MinMaxScaler().fit(scaler_df)
    if scaler == 'N':
        scaler = Normalizer().fit(scaler_df)
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
