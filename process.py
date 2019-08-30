import pandas as pd
from sklearn.preprocessing import Imputer,StandardScaler,MinMaxScaler,Normalizer


#change object column to dummy
def get_dummy(df):
    obj_list = list(df.select_dtypes("object"))
    dummy = pd.get_dummies(df[obj_list])
    df_new = df.drop(obj_list,axis=1)
    df_new = pd.concat([df_new,dummy],axis= 1)
    return df_new

#built scaler
def get_scaler(df,scaler = 'S'):
    scaler_list = list(df.select_dtypes(include = ["float64","int64"]))
    scaler_df = df[scaler_list]
    if scaler == 'S':
        scaler = StandardScaler().fit(scaler_df)
    if scaler == 'M':
        scaler = MinMaxScaler().fit(scaler_df)
    if scaler == 'N':
        scaler = Normalizer().fit(scaler_df)
    return scaler,scaler_list

#get scale value by chose scaler
def scale(df,scaler,scaler_list):
    tmp_df = df.drop(scaler_list,axis=1)
    scale_df = scaler.transform(df[scaler_list])
    scale_df = pd.DataFrame(scale_df,columns= scaler_list)
    scale_df = pd.concat([tmp_df,scale_df],axis=1)
    return scale_df
