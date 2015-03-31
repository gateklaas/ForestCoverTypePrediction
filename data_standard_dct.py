"""
Create files in a new format meant for Decision Trees
Shuffe rows
split up input file

Input file: train.csv
Output file: training_set.csv, cross_validation_set.csv, 
             test_set.csv and complete_set.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
wild_area_cols = ["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]
soil_type_cols = ["Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
real_value_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
train_set_size = 0.7 # 70% = train_set, rest = test_set

# load csv
data = pd.read_csv(data_path + "train.csv")
del data["Id"]

# change aspect
data["Aspect"] = (data["Aspect"] + 115) % 360
data["Aspect"].plot(kind="hist", title="Aspect")

# collapse Wilderness_Area column
data["Wilderness_Area"] = 0
for x in xrange(len(wild_area_cols)):
    data["Wilderness_Area"] = np.bitwise_or(data["Wilderness_Area"], data[wild_area_cols[x]] * (x + 1))
    del data[wild_area_cols[x]]

# collapse Soil_Type column
data["Soil_Type"] = 0
for x in xrange(len(soil_type_cols)):
    data["Soil_Type"] = np.bitwise_or(data["Soil_Type"], data[soil_type_cols[x]] * (x + 1))
    del data[soil_type_cols[x]]

# normalization
data[real_value_cols] = data[real_value_cols].astype(np.float)
std_scale = preprocessing.StandardScaler().fit(data[real_value_cols])
data_std = std_scale.transform(data[real_value_cols])
data[real_value_cols] = data_std

# cover_type column should be last
cover_type = data.pop("Cover_Type")
data["Cover_Type"] = cover_type

# shuffle rows
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index()
del data["index"]

# split data into three sets
n_samples = len(data)
test_start = (int) (n_samples * train_set_size)

train_set = data[0:test_start]
test_set = data[test_start:n_samples]

# create dct/ directory
if not os.path.exists(data_path + "dct/"):
    os.makedirs(data_path + "dct/")

# save csv
train_set.to_csv(data_path + "dct/training_set.csv")
test_set.to_csv(data_path + "dct/test_set.csv")
#data.to_csv(data_path + "dct/complete_set.csv")

print "success!"