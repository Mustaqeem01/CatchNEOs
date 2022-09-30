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
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("neo_v2.csv")
dl=df.mask(df["hazardous"]==False)
dk=df
dk=pd.concat([dk,dl,dl,dl,dl,dl,dl,dl,dl],axis=0)

dk=dk.sample(frac=1)
df=dk.dropna()
X = df[['est_diameter_min', 'est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude']]

y = df["hazardous"].replace([True, False],[1,0])
from sklearn.preprocessing import StandardScaler 

  

scalar = StandardScaler() 

  
# fitting 
#scalar.fit(df) 

#scaled_data = scalar.transform(df)

clf = LogisticRegression()
clf.fit(X, y)
#clf1 = PCA() 
#clf1.fit(X, y)
"""clf2 = SVC(kernel = 'linear', random_state = 0, C=1.0) 
clf2.fit(X, y)
clf5 = SVC(kernel = 'rbf', random_state = 0, C=1.0) 
clf5.fit(X, y)
clf6 = SVC(kernel = 'poly',degree=2, random_state = 0, C=1.0) 
clf6.fit(X, y)
clf7= SVC(kernel = 'poly',degree=3, random_state = 0, C=1.0) 
clf7.fit(X, y)
clf8= SVC(kernel = 'poly',degree=4, random_state = 0, C=1.0) 
clf8.fit(X, y)
clf9= SVC(kernel = 'poly',degree=5, random_state = 0, C=1.0) 
clf9.fit(X, y)
clf10= SVC(kernel = 'sigmoid', random_state = 0, C=1.0) 
clf10.fit(X, y)
clf11= SVC(kernel = 'precomputed', random_state = 0, C=1.0) 
clf11.fit(X, y)"""
clf3 = KNeighborsClassifier(2) 
clf3.fit(X, y)
clf5= KNeighborsClassifier(5) 
clf5.fit(X, y)
clf6 = KNeighborsClassifier(11) 
clf6.fit(X, y)
clf7= KNeighborsClassifier(17) 
clf7.fit(X, y)
clf8 = KNeighborsClassifier(21) 
clf8.fit(X, y)
clf9= KNeighborsClassifier(27) 
clf9.fit(X, y)
clf4 = RandomForestClassifier(n_estimators=100) 
clf4.fit(X, y)
import joblib

joblib.dump(clf, "clf.pkl")
#joblib.dump(clf2, "clf2.pkl")
#joblib.dump(clf1, "clf1.pkl")
joblib.dump(clf3, "clf3.pkl")
joblib.dump(clf4, "clf4.pkl")
joblib.dump(clf5, "clf5.pkl")
joblib.dump(clf6, "clf6.pkl")
joblib.dump(clf7, "clf7.pkl")
joblib.dump(clf8, "clf8.pkl")
joblib.dump(clf9, "clf9.pkl")
#joblib.dump(clf10, "clf10.pkl")
#joblib.dump(clf11, "clf11.pkl")

"""clf3 = KNeighborsClassifier(n_neighbours=2) 
clf3.fit(X, y)"""
"""
clf4.fit(X, y)
import joblib

joblib.dump(clf, "clf.pkl")
#joblib.dump(clf2, "clf2.pkl")
#joblib.dump(clf1, "clf1.pkl")
joblib.dump(clf3, "clf3.pkl")
joblib.dump(clf4, "clf4.pkl")
joblib.dump(clf5, "clf5.pkl")
joblib.dump(clf6, "clf6.pkl")
joblib.dump(clf7, "clf7.pkl")
joblib.dump(clf8, "clf8.pkl")
joblib.dump(clf9, "clf9.pkl")
#joblib.dump(clf10, "clf10.pkl")
#joblib.dump(clf11, "clf11.pkl")
""" 