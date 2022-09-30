import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score, classification_report, plot_confusion_matrix,plot_roc_curve, plot_precision_recall_curve, r2_score, log_loss

df=pd.read_csv("neo_v2.csv")

X = df[['est_diameter_min', 'est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude']]

y = df["hazardous"].replace([True, False],[1,0])
from sklearn.preprocessing import StandardScaler 

  

sc = StandardScaler()
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
estmin=0.1058168859
estmax=0.2366137501
rv=48425.8403287922
md=38355261.560761
am=22
x = pd.DataFrame([[estmin,estmax,rv,md,am]],columns = ['est_diameter_min','est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude'])
x= sc.fit_transform(x)
def probability_finder(y):
        p_prob=clf.predict_proba(y)
        return p_prob[0]
n=probability_finder(x)
print(n)
print(dir(clf))