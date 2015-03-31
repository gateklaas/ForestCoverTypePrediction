"""
Support Vector Machine with Gaussian/Linear/Sigmoid kernel

plot True-error/Dataset-size
print accuracy & f-score
print&save precision/recall-table to /reports
"""

from sklearn import metrics
from sklearn.svm import SVC
import os
import learning_curves as lc
import pandas as pd

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
X_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
y_column = "Cover_Type"

# load csv
train_set = pd.read_csv(data_path + "training_set.csv")
test_set = pd.read_csv(data_path + "test_set.csv")

# split
X_train = train_set[X_columns].values
y_train = train_set[y_column].values
X_test = test_set[X_columns].values
y_test = test_set[y_column].values

# Create report/ directory
if not os.path.exists(data_path + "report/"):
    os.makedirs(data_path + "report/")

# setup SVM with gaussian kernel & plot error
clf = SVC(kernel="rbf", C=50, gamma=0.2, cache_size=1000)
response = lc.test_error(clf, X_train, y_train, X_test, y_test)
lc.plot_test_error(*response, title="SVM with gaussian kernel")
clf.fit(X_train, y_train)
y_pred, y_true = y_test, clf.predict(X_test)
print "accuracy: %.5f" % metrics.accuracy_score(y_true, y_pred)
print "f-score: %.5f" % metrics.f1_score(y_true, y_pred)
lc.print_report(y_test, y_true)
lc.get_precision_recall_table(y_test, y_true).to_csv(data_path + "report/Gaussian SVM report.csv")

# setup SVM with linear kernel & plot error
clf = SVC(kernel="linear", C=3.5, cache_size=1000)
response = lc.test_error(clf, X_train, y_train, X_test, y_test)
lc.plot_test_error(*response, title="SVM with linear kernel")
y_pred, y_true = y_test, clf.predict(X_test)
print "accuracy: %.5f" % metrics.accuracy_score(y_true, y_pred)
print "f-score: %.5f" % metrics.f1_score(y_true, y_pred)
lc.print_report(y_test, y_true)
lc.get_precision_recall_table(y_test, y_true).to_csv(data_path + "report/Linear SVM report.csv")

# setup SVM with sigmoid kernel & plot error
clf = SVC(kernel="sigmoid", C=1000, gamma=0.001, cache_size=1000)
response = lc.test_error(clf, X_train, y_train, X_test, y_test)
lc.plot_test_error(*response, title="SVM with sigmoid kernel")
y_pred, y_true = y_test, clf.predict(X_test)
print "accuracy: %.5f" % metrics.accuracy_score(y_true, y_pred)
print "f-score: %.5f" % metrics.f1_score(y_true, y_pred)
lc.print_report(y_test, y_true)
lc.get_precision_recall_table(y_test, y_true).to_csv(data_path + "report/Sigmoid SVM report.csv")
