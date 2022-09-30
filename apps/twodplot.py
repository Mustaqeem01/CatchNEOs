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
	header(f"Statistical Exploratory Data Analysis of Previous Near Earth Objects with data visualization and linear regression(2D distributions and plots):")
	pn2=st.selectbox(f'Select a Distribution/Plot option',('Density Heatmap','Density Contour','Scatter Plot','Line Graph','Area Plot','Strip Chart','Regression Plot'))
	x2=st.selectbox(f'Select the feature for x axis',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	y2=st.selectbox(f'Select the feature for y axis',('est_diameter_min', 'est_diameter_max',
	       'relative_velocity', 'miss_distance', 'absolute_magnitude'))
	
	if st.button('Generate 2D Distribution or Plot'):
		if pn2=='Box Plot':
			fig=px.box(df, y=y2,x=x2, color='hazardous')
			st.plotly_chart(fig)
		if pn2=='Violin Plot':
			fig=px.violin(df, y=y2,x=x2, box=True, point='all',color='hazardous')
			st.plotly_chart(fig)
		if pn2=='Density Heatmap':
			fig=px.density_heatmap(df,x=x2,y=y2,marginal_x="histogram", marginal_y="histogram",facet_row='hazardous')
			st.plotly_chart(fig)
		if pn2=='Density Contour':
			fig=px.density_contour(df,x=x2,y=y2,marginal_x="histogram", marginal_y="histogram",color='hazardous')
			st.plotly_chart(fig)
		if pn2=='Scatter Plot':
			fig=px.scatter(df,x=x2,y=y2,color='hazardous',symbol='hazardous')
			st.plotly_chart(fig)
		if pn2=='Line Graph':
			fig=px.line(df,x=x2,y=y2,color='hazardous')
			st.plotly_chart(fig)
		if pn2=='Area Plot':
			fig=px.area(df,x=x2,y=y2,color='hazardous',line_group='hazardous')
			st.plotly_chart(fig)
		if pn2=='Strip Chart':
			fig=px.strip(df,x=x2,y=y2,color='hazardous')
			st.plotly_chart(fig)
		if pn2=='Regression Plot':
			fig=px.scatter(df,x=x2,y=y2,trendline='ols',color='hazardous',symbol='hazardous')
			st.plotly_chart(fig)