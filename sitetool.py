import json
import pandas as pd

def arrange_df(df,y,drops):
    """
    separate dataframe into X y

    Args:
    ----------
        df : dataframe
        y : target varable
        drops : columns to drop
    Return:
    ----------
        X : Train varable
        y : target varable
    """
    drops.append(y)
    X = df.drop(columns = drops)
    y = df[y]
    return X,y

def load_json(file,df = False,orient='split'):
    """
    load json file

    Args:
    ----------
        file : json file
        df : boolean transform to dataframe
        orient : orient for json
    Return:
    ----------
        jf : json
    """
    if df:
        jf = pd.read_json(file,orient=orient)
        return jf
    else:
        with open(file , 'r') as reader:
            jf = json.loads(reader.read())
        return jf
