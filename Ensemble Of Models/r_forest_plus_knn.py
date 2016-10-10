"""
This code generate 10 different submission files with varying weightage to the models.
"""

import pandas as pd
import numpy as np

def get_mapping():
    from sframe import SFrame as sf  # pandas dataframe is unable to understand type list in read_csv method
    mapping = sf.read_csv("../mapping_20604.csv", column_type_hints={"train_list1": list, "train_list2": list,
                                                                     "train_list3": list, "train_list4": list,
                                                                     "test_list": list})
    mapping = mapping.to_dataframe()  # now pandas dataframe will understand list
    return mapping


def get_corresponding_grid(index, mapping):
    neighboring_grid_blocks = [-103, -102, -101, -1, 0, 1, 103, 102, 101]
    columns = ["train_list1", "train_list2", "train_list3", "train_list4"]
    l = []
    for block in neighboring_grid_blocks:
        for col in columns:
            l = l + mapping[col].loc[index + block]

    return l


def result_maker(one_row):
    result = str(one_row[-1])+ " " + str(one_row[-2])+ " " + str(one_row[-3])
    return result


def calculate_distance(distances):
    return distances**-2


train = pd.read_csv("../engineered_train.csv")
test = pd.read_csv("../engineered_test.csv")
mapping = get_mapping()
result_df = pd.DataFrame(columns=["row_id"])

#Models
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="manhattan", n_jobs=2)
features_knn = ["x", "y", "hour_of_day", "day_of_week", "month_of_year", "month", "accuracy"]
feature_weights = [1, 1.86141237, 6.95271587, 7.74644896, 1.74877579, 1.87177237, 14.76085078]

from sklearn.ensemble import RandomForestClassifier
r_forest = RandomForestClassifier(n_estimators=40,min_samples_split=3,n_jobs=3)
features_r_forest = ["x", "y", "hour_of_day", "day_of_week", "month_of_year", "accuracy", "month"]

weights = np.arange(0,2,0.2)
indices = mapping.index
for i in indices:
    selected_test = test.ix[mapping.loc[i]["test_list"]]
    if len(selected_test) == 0:
        print "continued on -", i
        continue
    selected_train = train.ix[get_corresponding_grid(i, mapping)]
    a = selected_train["place_id"].value_counts()
    mask = (a[selected_train["place_id"].values] > 5).values
    selected_train = selected_train[mask]
    r_forest.fit(selected_train[features_r_forest],selected_train["place_id"])

    redundent_train1 = selected_train[selected_train["hour_of_day"]<4]
    redundent_train2 = selected_train[selected_train["hour_of_day"]>21]
    redundent_train1["hour_of_day"] = redundent_train1["hour_of_day"] +24
    selected_train = (selected_train.append(redundent_train1)).reset_index()
    redundent_train2["hour_of_day"] = redundent_train2["hour_of_day"] -24
    selected_train = (selected_train.append(redundent_train2)).reset_index()

    knn_1 = KNeighborsClassifier(n_neighbors=neighbors,n_jobs=3,p = 1,weights=calculate_distance,leaf_size=15)
    knn_1.fit(selected_train[features_knn]*feature_weights,selected_train["place_id"])

    forest_probs = r_forest.predict_proba(selected_test[features_r_forest])
    knn_1_probs = knn_1.predict_proba(selected_test[features_knn]*feature_weights)
    final_cols = ["row_id"]
    for weight in weights:
        predictions = forest_probs + weight*knn_1_probs
        colname = "p_id" +  str(weight)
        top_3_indices = np.argsort(predictions,axis = 1)[:,-3:]
        top_3 = knn_1.classes_[top_3_indices]
        top_3 = map(result_maker,top_3)
        selected_test[colname]= top_3
        final_cols.append(colname)

    new_df = selected_test[final_cols]
    result_df = result_df.append(new_df)
    print i

for weight2 in weights:
    colname2 = "p_id" +  str(weight2)
    result_df["place_id"] = result_df[colname2]
    result_df[["row_id","place_id"]].to_csv(colname2 + "complete.csv", index = False)
    print weight2
