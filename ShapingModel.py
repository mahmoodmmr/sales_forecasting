import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# %matplotlib inline
# import datetime
# import statsmodels.tsa.api as smt

def convertStringToDate():
    tempData = load_data()
    # data = data.copy()
    #data.SHOP_DATE = data.SHOP_DATE.apply(lambda x: "".join([str(x)[:4], "-", str(x)[4:6],"-", str(x)[6:8]]))
    tempData.date = tempData.date.apply(lambda x: "".join([str(x)[:4], "-", str(x)[4:6],"-", str(x)[6:8]]))
    data.sales = data.sales.apply(lambda x: x)
    print(data.date)
    #data = data.groupby('SHOP_DATE')['QUANTITY'].sum().reset_index()
    
    data = data.groupby('date')['sales'].sum().reset_index()

    # data.SHOP_DATE = pd.to_datetime(data.SHOP_DATE)
    data.to_csv('data/trainWallmart copy.csv')
    return data


# convertStringToDate(wallmartData)

def load_data():  
    return pd.read_csv('/data/trainWallmart copy.csv')  


