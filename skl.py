
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy.core.numeric import True_
from sklearn import metrics
#import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


df=pd.read_csv("neo_v2.csv")


X = df[['est_diameter_min', 'est_diameter_max',
       'relative_velocity', 'miss_distance', 'absolute_magnitude']]

y = df["hazardous"].replace([True, False],[1,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import NearestCentroid,RadiusNeighborsClassifier
print("done")
K= [] 

training = [] 

test = [] 

scores = {} 

  

for k in range(1, 5): 

    clf = RadiusNeighborsClassifier(radius=k) 

    clf.fit(X_train, y_train) 

  

    training_score = clf.score(X_train, y_train) 

    test_score = clf.score(X_test, y_test) 

    K.append(k) 

  

    training.append(training_score) 

    test.append(test_score) 

    scores[k] = [training_score, test_score]
    print(k,training_score,test_score)