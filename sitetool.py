import json
import pandas as pd


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
