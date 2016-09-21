""" some basic preprocessing on the time feature of the dataset.
There has been an assumption that the time feature in the dataset is in minutes.
For details about how this conclusion has been made, see competition Kernels on kaggle
"""


import pandas as pd
import numpy as np

def process_time(dataframe):
    dataframe["hour"] = dataframe["time"]//60
    dataframe["hour_of_day"] = dataframe["hour"]%24 + 1
    dataframe["day"] = dataframe["hour"]//24
    dataframe["day_of_year"] = dataframe["day"]%365
    dataframe["day_of_week"] = dataframe["day"]%7 + 1
    dataframe["month"] = dataframe["day"]//30 + 1
    dataframe["month_of_year"] = (dataframe["month"]-1)%12 + 1
    dataframe["sine"] = np.sin(2*np.pi*dataframe["hour_of_day"]/24)
    dataframe["cos"] = np.cos(2*np.pi*dataframe["hour_of_day"]/24)
    dataframe["year"] = dataframe["day"]//365 + 1
    return dataframe


train = pd.read_csv("train.csv")
engineered_train = process_time(train)
del train
engineered_train.to_csv("engineered_train.csv",index=False)

test = pd.read_csv("test.csv")
engineered_test = process_time(test)
del test
engineered_test.to_csv("engineered_test.csv",index = False)
