import pandas as pd
import numpy as np

""" Random forest on grids. The quality of result is limited due to the availability of ram.
"""

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


train = pd.read_csv("../engineered_train.csv")
test = pd.read_csv("../engineered_test.csv")
mapping = get_mapping()

features_r_forest = ["x", "y", "hour_of_day", "day_of_week", "month_of_year", "accuracy", "month"]
from sklearn.ensemble import RandomForestClassifier
r_forest = RandomForestClassifier(n_estimators=40,min_samples_split=3,n_jobs=3)


indices = mapping.index
for i in indices:
    selected_test = test.ix[mapping.loc[i]["test_list"]]
    if len(selected_test) == 0:
        print "continued on -", i
        continue
    selected_train = train.ix[get_corresponding_grid(i,mapping)]
    results = selected_train["place_id"]
    r_forest.fit(selected_train[features_r_forest],results)
    predictions = r_forest.predict_proba(selected_test[features_r_forest])
    top_3_indices = np.argsort(predictions,axis = 1)[:,-3:]
    top_3 = r_forest.classes_[top_3_indices]
    top_3 = map(result_maker,top_3)
    selected_test["place_id_nn"] = top_3
    test.set_value(mapping.loc[i]["test_list"],"place_id_nn",top_3)

test["place_id"] = test["place_id_nn"]
test[["row_id","place_id"]].to_csv("submission.csv", index= False)