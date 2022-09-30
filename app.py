import streamlit as st

from multiapp import MultiApp

import base64
import random
from apps import classify, onedplot, twodplot, thrdplot, polar, aneo, abdataset, eda, abproject

st.set_page_config(layout="wide")
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
@st.cache(persist= True)
def load():
    data= pd.read_csv("neo_v2.csv")
    return data


n=random.randrange(0,1)

def text1(url):
     st.markdown(f'<p style="background-color:#b3e7ff;color:#000000;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

@st.cache(persist=True)
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, X_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()


def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:14px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



def get_base64_vid(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data)

video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://drive.google.com/uc?export=download&id=18zi1-zkSHMs-skZ4xYPTvr4HR2pqSmTJ" type="video/mp4">
		  Your browser does not support HTML5 video.
		</video>
        """



@st.cache(persist=True)
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/gif;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache(persist=True)
def set_background_video(mp4_file):
    bin_str = get_base64_vid(mp4_file)
    page_bg_vid = '''
    <video style="position:fixed;right:0;bottom:0;min-width:100%;min-height:100%;" controls="false" autoplay loop muted>
    	<source type="video/mp4" src="data:video/mp4;base64,%s">
    <video>
    
    ''' % bin_str
    
    st.markdown(page_bg_vid, unsafe_allow_html=True)


n=15


if n==0:
	set_background('ab.gif')
if n==1:
	set_background_video('1.mp4')
if n==2:
	set_background_video('2.mp4')
if n==3:
	set_background_video('3.mp4')
if n==4:
	set_background_video('4.mp4')
header1("Analyzing Nearest Earth Objects and Discovering Their Risk")


if st.checkbox("Use Background Animation",False):
	st.markdown(video_html, unsafe_allow_html=True)


st.warning("Refreshing the page before navigating into another tool may emhance performance.")

header("Page Navigation:")
app = MultiApp()

app.add_app("About the Project", abproject.app)
app.add_app("About Near Earth Objects", aneo.app)
app.add_app("About the Dataset", abdataset.app)
app.add_app("Exploratory Data Analysis",eda.app)
app.add_app("Hazardous Near Earth Object Classifier by using different Machine Learning Models", classify.app)
app.add_app("Exploratory Near Earth Object Data Analysis and Visualization with 1D Distributions", onedplot.app)
app.add_app("Exploratory Near Earth Object Data Analysis and Visualization with 2D Distributions and Graphs", twodplot.app)
app.add_app("Exploratory Near Earth Object Data Analysis and Visualization with 3D Graphs", thrdplot.app)
app.add_app("Exploratory Near Earth Object Data Analysis and Visualization with Polar Graphs", polar.app)
app.run()