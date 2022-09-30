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
	header(f"Statistical Exploratory Data Analysis of Previous Near Earth Objects with data visualization(Polar Charts):")
	pn4=st.selectbox(f'Select a Distribution/Plot option',('Polar Scatter Plot','Polar Line Graph'))
	x4=st.selectbox(f'Select the feature for radius',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	y4=st.selectbox(f'Select the feature for theta',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	if st.button('Generate Polar Chart'):
		if pn4=='Polar Scatter Plot':
			fig=px.scatter_polar(df,r=x4,theta=y4,color='hazardous')
			st.plotly_chart(fig)
		if pn4=='Polar Line Graph':
			fig = px.line_polar(df, r=x4, theta=y4,color="hazardous", line_close=True,color_discrete_sequence=px.colors.sequential.Plasma_r,template="plotly_dark")
			st.plotly_chart(fig)