"""
Plot: True error/dataset size

Determine if machine learning parameters are properly adjusted (when error is low)
Determine if a bigger dataset is helpfull (if gap is big)
"""

from sklearn.svm import SVC
import learning_curves as lc
import pandas as pd

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
X_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
y_column = "Cover_Type"

# load csv
train_set = pd.read_csv(data_path + "training_set.csv")
test_set = pd.read_csv(data_path + "test_set.csv")

# select attributes
train_X = train_set[X_columns].values
train_y = train_set[y_column].values
test_X = test_set[X_columns].values
test_y = test_set[y_column].values

# support vector classification
clf = SVC(kernel="linear", C=3.5, cache_size=1000)
response = lc.test_error(clf, train_X, train_y, test_X, test_y)
lc.plot_test_error(*response, title="SVM with linear kernel")

clf = SVC(kernel="rbf", C=2.7, gamma=0.65, cache_size=1000)
response = lc.test_error(clf, train_X, train_y, test_X, test_y)
lc.plot_test_error(*response, title="SVM with gaussian kernel")
