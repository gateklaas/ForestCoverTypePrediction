"""
Create files in a new format meant for Support Vector Machines
Shuffe rows
split up input file

Input file: train.csv
Output file: training_set.csv, cross_validation_set.csv, 
             test_set.csv and complete_set.csv
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
real_value_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
train_set_size, crossv_set_size = 0.6, 0.2 # 60% = train_set, 20% = crossv_set, rest = test_set

# load csv
data = pd.read_csv(data_path + "train.csv")
del data["Id"]

# normalization
std_scale = preprocessing.StandardScaler().fit(data[real_value_cols])
data_std = std_scale.transform(data[real_value_cols])
data[real_value_cols] = data_std

# shuffle rows
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index()

# split data into three sets
n_samples = len(data)

train_end = (int) (n_samples * train_set_size)
crossv_end = train_end + (int) (n_samples * crossv_set_size)

train_set = data[0:train_end]
crossv_set = data[train_end:crossv_end]
test_set = data[crossv_end:n_samples]

# save csv
train_set.to_csv(data_path + "training_set.csv")
crossv_set.to_csv(data_path + "cross_validation_set.csv")
test_set.to_csv(data_path + "test_set.csv")
data.to_csv(data_path + "complete_set.csv")
