


import streamlit as st
import pandas as pd


def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

#@st.cache(persist=True)
def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


@st.cache(persist= True)
def load():
    data= pd.read_csv("neo_v2.csv")
    return data


#@st.cache(persist=True)
def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:22px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)



def app():
	import streamlit as st
	st.sidebar.header("Contributors")
	st.sidebar.write("Developer: Mustaqeem Morshed, 12th Grade, Dhaka College")
	st.sidebar.write("Coordinator: Jahin Farhan Jisan, 12th Grade, Dhaka College")
	st.sidebar.write("Tester: Akif Hossain Alif, 12th Grade, Dhaka College")
	st.sidebar.write("Developer Contact: fmustaqeem02@protonmail.com")
	header3(f"This project is an web application that analyzes and visualizes previous near earth objects data (collected from NASA API) with Exploratory Data Analysis and predicts if a new or existing near earth object is hazardous for our planet or not and also finds out the probability. There are more than 15 machine learning models available including Feedforward Artificial Neural Networks. Most of the models permit hyperparameters choise. Interactive feature selection, model analysis, prediction report and accuracy analysis has been developed in this project. There are more than 15 different kinds of data visualizations are available all of which are interactive 2D or 3D plots. Besides some Exploratory Data Analysis, we have always tried to keep the dataset updated. The data has been collected by an automated python script using https://api.nasa.gov/neo/rest/v2/. The script is frequently run by us and gets updated. In that script, we have used requests and multiprocessing library of python and extracted data from JSON and put that into a csv file. We want to thank stackoverflow, kaggle and github contributors because these sites helped us a lot for making that script. NASA API includes many RESTful requests each serving a different purpose. However, the API restricts the number of days to be less than or equal to 10 for which the request is sent. Hence, we had send multiple requests to the API each containing a different starting date. Further, we have used multiple processes that run on multiple cores. Here are some tools we have been using for this project: Python, JavaScript, Plotly.js, Scikit Learn, Streamlit etc. This application is mostly written with Python. And for generating 2D and 3D interactive graphs and charts, we have used plotly.js. As we have used some complex algprithms for some models, sometimes the application slows down. To solve this, the application uses cache to improve performance.")