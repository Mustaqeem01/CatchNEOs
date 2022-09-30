import streamlit as st
import pandas as pd



def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:22px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def text1(url):
     st.markdown(f'<p style="background-color:#b3e7ff;color:#000000;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)  




def load():
    data= pd.read_csv("neo_v2.csv")
    return data


def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        header3("Confusion Matrix")
        plot_confusion_matrix(clf, X_test, y_test)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        header3("ROC Curve")
        plot_roc_curve(clf, X_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        header3("Precision-Recall Curve")
        plot_precision_recall_curve(clf, X_test, y_test)
        st.pyplot()



def dtree(X,y, features, sfeatures,metrics):
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import precision_score,recall_score, classification_report, plot_confusion_matrix,plot_roc_curve, plot_precision_recall_curve, r2_score, log_loss, cohen_kappa_score, balanced_accuracy_score
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1, random_state=0)
	X_train= sc.fit_transform(X_train)
	X_test= sc.fit_transform(X_test)
	clf=DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	x = pd.DataFrame([sfeatures],columns = features)
	x= sc.fit_transform(x)
	prediction = clf.predict(x)[0]
	if 'predict_proba' in dir(clf):
		p_prob=clf.predict_proba(x)[0]
		hprob=p_prob[1]
		nhprob=p_prob[0]
		st.success(f"Probablility of the neo being not hazardous: {nhprob}")
		st.success(f"Probablility of the neo being hazardous: {hprob}")
	y_pred=clf.predict(X_test)
	bac=balanced_accuracy_score(y_test,y_pred)
	sccr=clf.score(X_test,y_test)
	r2=r2_score(y_test, y_pred)
	ll=log_loss(y_test,y_pred)
	cks=cohen_kappa_score(y_test,y_pred)
	cr=classification_report(y_test, y_pred,output_dict=True)
	st.success(f"R squared score: {r2}")
	st.success(f"Log loss: {ll}")
	st.success(f"Balanced accuracy score:{bac}")
	st.success(f"Cohen Kappa Score: {cks}")
	header("Classification Report")
	st.dataframe(pd.DataFrame(cr).transpose())
	plot_metrics(metrics)
	if prediction==0:
		p="not hazardous. We are about safe from it!"
	if prediction==1:
		p="hazardous for our planet!!!! We must try to  protect Earth immediately from it!!!!"
	st.success(f"Prediction Result: This near earth object is {p}")