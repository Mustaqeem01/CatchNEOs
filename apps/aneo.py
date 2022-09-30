import streamlit as st
import pandas as pd



def header1(url):
     st.markdown(f'<p style="color:#b366ff;font-size:40px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


def header(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header3(url):
     st.markdown(f'<p style="color:#ffbf80;font-size:22px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)




def app():
	import pandas as pd
	import streamlit as st
	header(r"A glimpse into What is Near Earth Objects (Source: Nasa):")
	video_file = open('myvideo.mp4', 'rb')
	video_bytes = video_file.read()
	st.video(video_bytes)
	st.write('')
	st.write("")
	header("What are near earth objects?")
	header3(f"From time to time, we have all seen stories in the Press about Near Earth Objects that are about to hit the Earth on some date in the not-too-distant future.The nearest earth objects are mainly come from asteroid main belt which is in between Mars and Jupiter and it can happen because of a collision or because of a interaction.An NEO has an orbit which takes it within 1.3 AU of the Sun. The largest known NEO is over 40 km in diameter. Astronomers have found thousands of NEOs with diameters larger than 1 km. Anything smaller is too hard to see.The aim here is to briefly describe what the normal practice is when a NEO is discovered and what part the we might  see to play in this process.  In fact, the basic procedure is the same whether the discovery is of a comet, a Kuiper belt object, a Main belt asteroid or any other minor body in the Solar System. A large NEO could collide with the Earth in the future. The impact would cause widespread damage. The most dangerous, or hazardous, objects are the biggest and closest ones to the Earth. A hazard it does not mean it will hit the Earth. We need to spot these hazards early to do this. By seeing them early we can prevent a disaster. Remember that a large asteroid colliding with the Earth is not likely!")
	st.write("")
	st.write("")
	header("Near Earth object dataset:")
	st.dataframe(pd.read_csv("neo_v2.csv"))