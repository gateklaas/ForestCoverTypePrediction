"""
Create a submission file

Input files: train.csv and test.csv
Output file: test_submission.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics

print "Initialize"
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
real_value_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
X_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
y_column = "Cover_Type"
submission_cols = ["Id", "Cover_Type"]
n_test_samples = 565892
chunksize = 100000

print "Init machine learning"
clf = SVC(kernel="rbf", C=6, gamma=0.55, cache_size=1000)

print "Load train data"
data = pd.read_csv(data_path + "train.csv")

print "Change aspect"
data["Aspect"] = (data["Aspect"] + 115) % 360

print "Normalize train data"
std_scale = preprocessing.StandardScaler()
std_scale.fit(data[real_value_cols])
data[real_value_cols] = std_scale.transform(data[real_value_cols])

print "Test machine learning accuracy"
train_X, test_X, train_y, test_y = train_test_split(data[X_columns].values, data[y_column].values)
clf.fit(train_X, train_y)
print "accuracy_score: %.5f" % metrics.accuracy_score(test_y, clf.predict(test_X))

print "Setup machine learning for real"
clf.fit(data[X_columns].values, data[y_column].values)

print "Load test data"
reader = pd.read_csv(data_path + "test.csv", chunksize=chunksize)
n_chunks = int (n_test_samples / chunksize + 1)

print "Create test/ directory for chucks"
if not os.path.exists(data_path + "test/"):
    os.makedirs(data_path + "test/")

i = 0
for data in reader:
    i += 1
    print "Test data chunk %d/%d" % (i, n_chunks)
    
    print " change aspect"
    data["Aspect"] = (data["Aspect"] + 115) % 360

    print " normalize"
    data[real_value_cols] = std_scale.transform(data[real_value_cols].astype(np.float))
    
    print " predict"
    data[y_column] = clf.predict(data[X_columns].values)
    
    print " save to test/"
    data[submission_cols].to_csv(data_path + "test/test_submission_chunk_%d.csv" % i)

print "Merge chunks"
fout = open(data_path + "test_submission.csv", "w+")
for i in range(1, n_chunks + 1):
    print "Chunk %d/%d merge" % (i, n_chunks)
    f = open(data_path + "test/test_submission_chunk_" + str(i) + ".csv")
    if i > 1:
        f.next()
    for line in f:
         fout.write(line)
    f.close()
fout.close()

print "Calcalculate accuracy"
data1 = pd.read_csv(data_path + "test_submission_100%_accurate.csv")
data2 = pd.read_csv(data_path + "test_submission.csv")
print "Accuracy: %f" % metrics.accuracy_score(data1[y_column].values, data2[y_column].values)

print "Success!"