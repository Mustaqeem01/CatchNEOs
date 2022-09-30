import streamlit as st
import pandas as pd

def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:14px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


@st.cache(persist= True)
def load():
    data= pd.read_csv("neo_v2.csv")
    return data




def app():
	import plotly.express as px
	df=load()
	header(f"Statistical Exploratory Data Analysis of Previous Near Earth Objects with data visualization(3D plots):")
	pn3=st.selectbox(f'Select a Distribution/Plot option',('Scatter Plot','Line Graph'))
	x3=st.selectbox(f'Select the feature for X axis',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	y3=st.selectbox(f'Select the feature for Y axis',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	z3=st.selectbox(f'Select the feature for z axis',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	if st.button('Generate 3D Plot'):
		if pn3=='Scatter Plot':
			fig=px.scatter_3d(df,x=x3,y=y3,z=z3,color='hazardous',symbol='hazardous')
			st.plotly_chart(fig)
		
		if pn3=='Line Graph':
			fig=px.line_3d(df,x=x3,y=y3,z=z3,color='hazardous')
			st.plotly_chart(fig)