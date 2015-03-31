"""
Create files in a new format meant for Support Vector Machines
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
from matplotlib import pylab, mlab, pyplot
plt = pyplot

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
real_value_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
train_set_size = 0.7 # 70% = train_set, rest = test_set

# load csv
data = pd.read_csv(data_path + "train.csv")
del data["Id"]

# change aspect
data["Aspect"] = (data["Aspect"] + 115) % 360
data["Aspect"].plot(kind="hist", title="Aspect")

# normalization
data[real_value_cols] = data[real_value_cols].astype(np.float)
std_scale = preprocessing.StandardScaler().fit(data[real_value_cols])
data_std = std_scale.transform(data[real_value_cols])
data[real_value_cols] = data_std

# shuffle rows
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index()
del data["index"]

# split data into three sets
n_samples = len(data)
test_start = (int) (n_samples * train_set_size)

train_set = data[0:test_start]
test_set = data[test_start:n_samples]

# create svm/ directory
if not os.path.exists(data_path + "svm/"):
    os.makedirs(data_path + "svm/")

# save csv
train_set.to_csv(data_path + "svm/training_set.csv")
test_set.to_csv(data_path + "svm/test_set.csv")
#data.to_csv(data_path + "svm/complete_set.csv")

print "success!"