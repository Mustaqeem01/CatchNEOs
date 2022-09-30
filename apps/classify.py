import streamlit as st
import pandas as pd
from sklearn.metrics import precision_score,recall_score, classification_report, plot_confusion_matrix,plot_roc_curve, plot_precision_recall_curve, r2_score, log_loss, cohen_kappa_score, balanced_accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)
class_names=["Hazardous","Not Hazardous"]
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



def modify_data(data):
	dl=data.mask(data["hazardous"]==False)
	dk=data
	dk=pd.concat([dk,dl,dl,dl,dl,dl,dl,dl,dl],axis=0)
	dk=dk.sample(frac=1)
	dn=dk.dropna()
	return dn





def app():
    from numpy.core.numeric import True_
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    data=load()
    df=data
    if st.sidebar.checkbox("Display Dataset", False):
        header3("Near Earth Objects Dataset:")
        st.dataframe(data)
    dl=df.mask(data["hazardous"]==False)
    dk=df
    dk=pd.concat([dk,dl,dl,dl,dl,dl,dl,dl,dl],axis=0)
    dk=dk.sample(frac=1)
    df=dk.dropna()
    df=df.sample(axis=0, frac=1)
    st.sidebar.subheader("Choose Model")
    classifier = st.sidebar.selectbox( 'Classifiers',("Logistic Regression", "Feedforward Artificial Neural Network /(Multi Layer Perceptron/)","Ridge Classifier","Perceptron", "Decision Tree", "Adaboost", "Gradient Boosting", "Histogram Based Gradient Boosting", "Stochastic Gradient Descent Classifier","KNN", 'Random Forest','Nearest Centroid Classifier','Linear Discriminant Analysis','Quadratic Discriminant Analysis','Passive Aggressive Classifier','Extra Tree Classifier'))
    features=st.sidebar.multiselect('Feature Selection',['est_diameter_min','est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude'])
    X = df[features] if features else df[['est_diameter_min','est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude']]
    y = df["hazardous"].replace([True, False],[1,0])
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1, random_state=0)
    
    
    def scale_df(X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test
    
    
        
    st.warning("Please refresh the page before running another model after a model to get better performance. Do not leave feature selection option empty")
    header("Enter the following details of any near earth object to discover if it is hazardous for our planet or not::")
    st.warning("Usage of this tool:\nEnter following details and choose any model to classify any near earth object. Do not leave any field blank or set to 0.")
    header3(r"Enter Estimated Diameter (minimum) in Kilometres:")
    estmin= st.number_input(r'Enter Estimated Diameter (minimum) in Kilometres:')
    header3(r"Enter Estimated Diameter (maximum) in Kilometres:")
    estmax= st.number_input(r'Enter Estimated Diameter (maximum) in Kilometres:')
    header3(r"Enter the velocity in kilometres per hour of the nearest earth object relative to Earth :")
    rv=st.number_input(r'Enter the velocity of the nearest earth object relative to Earth:')

    header3(r"Enter miss distance of the nearest earth object in Kilometres:")
    md=st.number_input(r"Enter miss distance of the nearest earth object in Kilometres:")

    header3(r"Enter absolute magnitude of the nearest earth object in Kilometres:")
    am=st.number_input(r'Enter absolute magnitude of the nearest earth object in Kilometres:')
    
    
    
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        mx_iter = st.sidebar.slider("Maximum iterations", 10, 500, key="max_iter_lr")
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparameters")
        n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    if classifier == "KNN":
        st.sidebar.subheader("Hyperparameters")
        k=st.sidebar.slider("Value of K", 1, 500, key="k")
        wht=st.sidebar.radio('Weights',('uniform','distance'),key="knn")
        alg=st.sidebar.radio('Algorithm',('auto','ball_tree','kd_tree','brute'),key='kn')
    if classifier == 'Stochastic Gradient Descent Classifier':
        lf=st.sidebar.radio("Loss Function",('hinge','log_loss','modified_huber','log','huber','squared_hinge',"epsilon_insensitive",'squared_epsilon_insensitive','perceptron'), key='lf')
        ma_iter = st.sidebar.slider("Maximum iterations", 10, 500, key="max_iter_sgd")
    if classifier=='Ridge Classifier':
        st.sidebar.write("")
    if classifier=='Passive Aggressive Classifier':
        lpac=st.sidebar.radio("Loss Function",('hinge','squared_hinge',"epsilon_insensitive",'squared_epsilon_insensitive'),key="lpac")
    if classifier=='Decision Tree':
        st.sidebar.write()
    if classifier=='Adaboost':
        st.sidebar.subheader("Hyperparameters")
        ne=st.sidebar.slider("The number of estimators", 1, 100, key="nea")
    if classifier=='Gradient Boosting':
        st.sidebar.subheader("Hyperparameters")
        ne=st.sidebar.slider("The number of estimators", 1, 100, key="neg")
        md=st.sidebar.slider("Maximum Depth", 1, 15, key="md")
    if classifier=='Histogram based Gradient Boosting':
        st.sidebar.subheader("Hyperparameters")
        miiiii=st.sidebar.slider("Maximum Iterations", 1, 100, key="miiiii")
    if classifier=='Feedforward Artificial Neural Network /(Multi Layer Perceptron/)':
        st.sidebar.subheader("Hyperparameters")
        svr=st.sidebar.radio("Solver",('lbfgs','adam','sgd'),key='slvrmlp')
        lr=st.sidebar.radio('Learning Rate',('constant','adaptive','invscaling'))
        act=st.sidebar.radio('Activation method',('tanh','relu','logistic','identity'))
        mt=st.sidebar.slider("Maximum Iteration",1,15,key='mt')
    if classifier=='Bernoulli Naive Bayes':
        st.sidebar.subheader("Hyperparameters")
        alphab=st.sidebar.slider('Alpha',0,5,key='ab')
    if classifier=='Gaussian Naive Bayes':
        st.write("")
    if classifier=='Nearest Centroid Classifier':
        st.sidebar.subheader("Hyperparameters")
        mtr=st.sidebar.radio("Metric",('euclidean','manhattan'))
    if classifier=='Linear Discriminant Analysis':
        st.sidebar.subheader("Hyperparameters")
        slvr=st.sidebar.radio('Solver',('svd','lsqr','eigen'),key='slvr')
    if classifier=='Quadratic Discriminant Analysis':
        st.sidebar.write('')
    if classifier=='Perceptron':
        st.sidebar.subheader("Hyperparameters")
        mxit=st.sidebar.slider('Maximum Iteration',1,30)
        
        
        
    
    
    metrics = st.sidebar.multiselect("Metrices to Plot:",("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    x=[]
    
    def predictor(classifier='Logistic Regression'):
        global x
        global X_train
        global X_test
        global y_train
        global y_test
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        
        X = df[features] if features else df[['est_diameter_min','est_diameter_max','relative_velocity', 'miss_distance', 'absolute_magnitude']]
        y = df["hazardous"].replace([True, False],[1,0])
        X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1, random_state=0)
        X_train, X_test=scale_df(X_train, X_test)
        sfeatures=[]
        if 'est_diameter_min' in features:
            sfeatures.append(estmin)
        if 'est_diameter_max' in features:
            sfeatures.append(estmax)
        if 'relative_velocity' in features:
            sfeatures.append(rv)
        if 'miss_distance' in features:
            sfeatures.append(md)
        if 'absolute_magnitude' in features:
            sfeatures.append(am)
        
        if 1:
            if classifier=="Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="KNN":
                from sklearn.neighbors import KNeighborsClassifier
                clf=KNeighborsClassifier(k,weights=wht,algorithm=alg)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Stochastic Gradient Descent Classifier":
                from sklearn.linear_model import SGDClassifier
                clf=SGDClassifier(loss=lf, max_iter=ma_iter)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Ridge Classifier":
                from sklearn.linear_model import RidgeClassifier
                clf=RidgeClassifier()
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            
            elif classifier=="Adaboost":
                from sklearn.ensemble import AdaBoostClassifier
                clf=AdaBoostClassifier(n_estimators=ne)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                clf=GradientBoostingClassifier(n_estimators=ne,max_depth=md)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Histogram Based Gradient Boosting":
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf=HistGradientBoostingClassifier()
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                clf=DecisionTreeClassifier()
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                clf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Gaussian Naive Bayes':
                from sklearn.naive_bayes import GaussianNB
                clf=GaussianNB()
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Bernoulli Naive Bayes':
                from skearn.naive_bayes import  BernoulliNB
                clf=BernoulliNB(alpha=alphab)
                clf.fit(X_train, y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Feedforward Artificial Neural Network /(Multi Layer Perceptron/)':
                from sklearn.neural_network import MLPClassifier
                clf=MLPClassifier(activation=act,solver=svr,learning_rate=lr,max_iter=mt,early_stopping=True)
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=="Nearest Centroid Classifier":
                from sklearn.neighbors import NearestCentroid
                clf=NearestCentroid(metric=mtr)
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Linear Discriminant Analysis':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf=LinearDiscriminantAnalysis(solver=slvr)
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Quadratic Discriminant Analysis':
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                clf=QuadraticDiscriminantAnalysis()
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Perceptron':
                from sklearn.linear_model import Perceptron
                clf=Perceptron()
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Passive Aggressive Classifier':
                from sklearn.linear_model import PassiveAggressiveClassifier
                clf=PassiveAggressiveClassifier(loss=lpac)
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            elif classifier=='Extra Tree Classifier':
                from sklearn.tree import ExtraTreeClassifier
                clf=ExtraTreeClassifier()
                clf.fit(X_train,y_train)
                x = pd.DataFrame([sfeatures],columns = features)
                x= sc.fit_transform(x)
                prediction = clf.predict(x)[0]
            else:
                prediction=0
           
            cnf=clf
            if 'predict_proba' in dir(cnf):
                p_prob=clf.predict_proba(x)[0]
                hprob=p_prob[1]
                nhprob=p_prob[0]
                st.success(f"Probablility of the neo being not hazardous: {nhprob}")
                st.success(f"Probablility of the neo being hazardous: {hprob}")
            return clf, prediction
        
    prediction=5
    
        
    sac=st.sidebar.button("Submit and Classify")
    if sac:
        clf, prediction=predictor(classifier) 
    try:   
        if prediction==0:
            p="not hazardous. We are about safe from it!"
        if prediction==1:
            p="hazardous for our planet!!!! We must try to  protect Earth immediately from it!!!!"
        st.success(f"Prediction Result: This near earth object is {p}")
    except:
        st.write('')
        
        
    
        
    
    
    
    def prediction_test(X_test):
        y_pred=clf.predict(X_test)
        return y_pred
     
        
    try:   
        y_pred=prediction_test(X_test)
        
        r2=r2_score(y_test, y_pred)
        st.success(f"R squared score: {r2}")
        ll=log_loss(y_test,y_pred)
        st.success(f"Log loss: {ll}")
        bac=balanced_accuracy_score(y_test,y_pred)
        st.success(f"Balanced accuracy score:{bac}")
        cks=cohen_kappa_score(y_test,y_pred)
        st.success(f"Cohen Kappa Score: {cks}")
        header("Classification Report")
        cr=classification_report(y_test, y_pred,output_dict=True)
        st.dataframe(pd.DataFrame(cr).transpose())
        header("Metrics:")
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
        plot_metrics(metrics)
    except:
        st.write('')