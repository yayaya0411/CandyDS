import json
import pandas as pd

def arrange_df(df,y,drops):
    """ separate dataframe into X y
    df : dataframe
    y : target varable
    drops : columns to drop
    """
    drops.append(y)
    X = df.drop(columns = drops)
    y = df[y]
    return X,y

def load_json(file,df = False,orient='split'):
    """ load json file
    file : json file
    df : boolean transform to dataframe
    orient : orient for json
    """
    if df:
        jf = pd.read_json(file,orient=orient)
        return jf
    else:
        with open(file , 'r') as reader:
            jf = json.loads(reader.read())
        return jf
