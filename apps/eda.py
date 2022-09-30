
import streamlit as st
import pandas as pd

def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


@st.cache(persist= True)
def load():
    data= pd.read_csv("neo_v2.csv")
    return data




def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:14px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)




def app():
	from PIL import Image
	from pandas_profiling import ProfileReport
	from streamlit_pandas_profiling import st_profile_report
	df=load()
	header(f"Grouping by(Mean) Hazardous column:")
	
	
	def gby(data):
		return data.groupby(['hazardous']).agg('mean')
	
	st.dataframe(gby(df))
	
	
	st.warning("Generating the profile report may take few minutes.")
	
	def profiler(data):
		return ProfileReport(data, explorative=True)
	pr = profiler(df)
	header("Profile report")
	st_profile_report(pr)