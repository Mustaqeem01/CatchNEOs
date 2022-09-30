
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
     st.markdown(f'<p style="color:#ffbf80;font-size:22px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)




def app():
	import pandas as pd
	from PIL import Image
	import streamlit as st
	from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
	header(f"Near Earth object dataset (made using api.nasa.gov):")
	data=load()
	
	gb = GridOptionsBuilder.from_dataframe(data)
	gb.configure_pagination(paginationAutoPageSize=True)
	gb.configure_side_bar() 
	gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
	gridOptions = gb.build()
	
	grid_response = AgGrid(data,gridOptions=gridOptions,data_return_mode='AS_INPUT',update_mode='MODEL_CHANGED',fit_columns_on_grid_load=False,theme='dark',enable_enterprise_modules=True,height=350,width='100%',reload_data=True)
	data = grid_response['data']
	selected = grid_response['selected_rows'] 
	dn= pd.DataFrame(selected)
	st.write("")
	st.write("")
	header("Column Descriptions")
	st.dataframe(pd.read_csv('columns.csv'))
	st.write("")
	st.write("")
	header("Informations about the Dataset")
	
	
	@st.cache(persist=True)
	def info(data):
		return pd.read_table(data)
	st.dataframe(info("df_info.txt"))
	st.write("")
	st.write("")
	header("Dataset Description:")
	
	
	@st.cache(persist=True)
	def desc(data):
		return data.describe()
	st.dataframe(desc(data))
	st.write("")
	st.write("")
	header("Correlations in the dataset:")
	
	@st.cache(persist= True)
	def cor(data):
		return data.corr()
	st.dataframe(cor(data))
	image = Image.open('hm.jpg')
	st.image(image, caption="Correlations in the dataset")
	st.write("")
	st.write("")
	header("How we have collected the data:")
	header3(f"The data has been collected by an automated python script using https://api.nasa.gov/neo/rest/v2/. The script is frequently run by us and gets updated. In that script, we have used requests and multiprocessing library of python and extracted data from JSON and put that into a csv file. We want to thank stackoverflow, kaggle and github contributors because these sites helped us a lot for making that script. NASA API includes many RESTful requests each serving a different purpose. However, the API restricts the number of days to be less than or equal to 10 for which the request is sent. Hence, we had send multiple requests to the API each containing a different starting date. Further, we have used multiple processes that run on multiple cores.")