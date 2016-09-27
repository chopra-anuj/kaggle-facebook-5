"""This code helps to find out feature multipliers for KNN.
This is shown using some features derived by me but this method can be extended for other features as well.
One needs to derive his own features and then apply similar approach to get the correct weights.
This code has been discussed here:
https://www.kaggle.com/chopra/facebook-v-predicting-check-ins/logistic-regression-to-find-knn-weights/code
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


def read_prepare_data(path):
    recent_train = pd.read_csv(path)
    # select a single x_y_grid at random
    recent_train = recent_train[
        (recent_train["x"] > 4500) & (recent_train["x"] < 5000) & (recent_train["y"] > 2000) & (recent_train["y"] < 2300)]
    # derive some features
    recent_train["x"], recent_train["y"] = recent_train["x"] * 1000, recent_train["y"] * 1000
    recent_train["hour"] = recent_train["time"] // 60
    recent_train["hour_of_day"] = recent_train["hour"] % 24 + 1
    recent_train["day"] = recent_train["hour"] // 24
    recent_train["day_of_week"] = recent_train["day"] % 7 + 1
    recent_train["month"] = recent_train["day"] // 30 + 1
    recent_train["month_of_year"] = (recent_train["month"] - 1) % 12 + 1
    recent_train["accuracy"] = np.log(recent_train["accuracy"])
    print("recent_train created")
    return recent_train


