""" formation of grids. The data provided by facebook consists of checkins over 10Km x 10Km area
In this script, the dataset is broken into smaller grids of size 50m x 100m. A padding is also introduced
at all the edges of 10Km square. This padding enables to add margins to individual grids (randomforest.py)
"""

import pandas as pd
import numpy as np

""" generates row_index for specific x,y value"""


def loc_generator(x, y):
    location = int((y / 50) + 1) * 102 + int((x / 100) + 1)
    return location


""" generates column_name for specific time value"""


def col_generator(time):
    if time < 200000:
        return "train_list1"
    if time < 400000:
        return "train_list2"
    if time < 600000:
        return "train_list3"
    else:
        return "train_list4"


def add_corresponding_row_col(dataframe):
    # convert Km to m
    dataframe["x"] = dataframe["x"] * 999.99
    dataframe["y"] = dataframe["y"] * 999.99

    dataframe["loc"] = dataframe.apply(lambda x: loc_generator(x["x"], x["y"]), axis=1)
    dataframe["col"] = dataframe["time"].apply(col_generator)
    return dataframe


""" This dataframe maps each row of train/test file to it's corresponding grid. The number of rows in this dataframe
is equal to the number of grids with padding and columns correspond to the time feature in train/test.
Each cell of this dataframe is a list which contains the indices of samples of train/test which fall in that grid in
that time.
"""
def create_new_mapping():
    ind = np.arange(20604)
    col = ["x", "y", "train_list1", "train_list2", "train_list3", "train_list4", "test_list"]
    mapping = pd.DataFrame(index=ind, columns=col)
    mapping["train_list1"] = [[] for i in range(20604)]
    mapping["train_list2"] = [[] for i in range(20604)]
    mapping["train_list3"] = [[] for i in range(20604)]
    mapping["train_list4"] = [[] for i in range(20604)]
    mapping["test_list"] = [[] for i in range(20604)]
    return mapping


def fill_mapping(mapping,dataframe):
    index = dataframe.index
    for i in index:
        elements = dataframe.loc[i]
        row = elements["loc"]
        column = elements["col"]
        mapping.loc[row][column].append(i)
    return mapping



mapping = create_new_mapping()

train = pd.read_csv("train.csv")
train = train.drop(["row_id", "accuracy"], axis=1)
train = add_corresponding_row_col(train)

test = pd.read_csv("test.csv")
test = test.drop(["row_id", "accuracy"], axis=1)
train = add_corresponding_row_col(train)

mapping = fill_mapping(mapping,train)
mapping = fill_mapping(mapping,test)

mapping.to_csv("mapping_20604.csv",index=False)