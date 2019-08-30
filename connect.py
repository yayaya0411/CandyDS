import pandas as pd
import pyodbc


def loadsql(query,conn,chunk=None):
    if chunk == None:
        df = pd.read_sql(quedry,conn)
        print('query load finish')
        return df
    else:
        df = pd.DataFrame()
        i=0
        for chunk_row in pd.read_sql(query,conn,chunksize = chunk):
            i += chunk
            print("read  " + str(i) + " rows")
            df = df.append(chunk_row)
        print('query load finish')
        return df
