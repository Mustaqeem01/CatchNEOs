
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
	dn=df
	header(f"Statistical Exploratory Data Analysis of Previous Near Earth Objects with data visualization(1D distributions):")
	st.warning("If histogram is selected enter bins number, else simply ignore it")
	pn1=st.selectbox('Select a Distribution option',('Histogram','ECDF'))
	x1=st.selectbox(f'Select the feature for x axis(for histogram) or y axis(for box plot or violin plot)',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	header3('Enter bins number for Histogram:')
	
	bin=st.number_input('')
	
	
	
	
	g1d=st.button('Generate 1D Distribution')
	
	if g1d:
		if pn1=='Histogram':
			fig = px.histogram(df, x=x1, marginal="rug",nbins=int(bin),color="hazardous")
			df=dn
			st.plotly_chart(fig)
		if pn1=='Box Plot':
			fig = px.box(df, y=x1, color='hazardous')
			df=dn
			st.plotly_chart(fig)
		if pn1=='Violin Plot':
			fig=px.violin(df, y=x1, box=True, point='all',color='hazardous')
			df=dn
			st.plotly_chart(fig)
		if pn1=='ECDF':
			fig=px.ecdf(df,x=x1,color='hazardous',ecdfnorm=None)
			df=dn
			st.plotly_chart(fig)