import pandas as pd

df=pd.read_csv("neo_v2.csv")
###print(dir(df))
"""
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
with open("df_info.txt", "w", encoding="utf-8") as f:
    f.write(s)
    """
#df['hazardous']=df['hazardous'].astype("object")
#print(pd.read_table("df_info.txt"))
#print(f"Dataset Unique Values:{df['name'].value_counts()}")
print(df.shape)
print(df.size)
print(pd.DataFrame(df.groupby(df['hazardous'])))

