"""
Support Vector Machine with Sigmoid kernel

plot C and gamma
"""

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import learning_curves as lc
import pandas as pd

# constants
data_path = "C:/Users/klaas/Documents/School/Vrije universiteit/Machine learning/project/csv/"
X_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
y_column = "Cover_Type"
k = 10

# load csv
dataset = pd.read_csv(data_path + "training_set.csv")

# limit dataset size
#dataset = dataset[:1000]

# split
kf = KFold(len(dataset), n_folds=k)
X = dataset[X_columns].values
y = dataset[y_column].values

# setup SVM, plot C and gamma
clf = SVC(kernel="sigmoid", C=1000, gamma=0.001, cache_size=1000)
response = lc.crossv_gamma_c(clf, X, y, kf, n_subsets=10)
lc.plot_crossv_gamma_c(*response, title="Parameters of SVM with sigmoid kernel")

# setup SVM, plot gamma
#clf = SVC(kernel="sigmoid", C=1000, gamma=0.001, cache_size=1000)
#response = lc.crossv_gamma(clf, X, y, kf)
#lc.plot_crossv_gamma(*response, title="SVM with sigmoid kernel")

# setup SVM, plot C
#clf = SVC(kernel="sigmoid", C=1000, gamma=0.001, cache_size=1000)
#response = lc.crossv_c(clf, X, y, kf)
#lc.plot_crossv_c(*response, title="SVM with sigmoid kernel")
