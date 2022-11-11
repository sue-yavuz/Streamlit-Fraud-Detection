import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import xgboost as xgb
import pickle
# pd.set_option('display.float_format', lambda x: '%.3f'%x)

st.set_page_config(page_title="Credit Card Fraud Detection App", page_icon="💰",
                   layout='centered', initial_sidebar_state='expanded')


model = pickle.load(open('Fraud_Detection_xgb.pkl', 'rb'))
# df
import zipfile

zf = zipfile.ZipFile('./creditcard.zip') 
# if you want to see all files inside zip folder
#print(zf.namelist())

# now read your csv file 
df = pd.read_csv(zf.open('creditcard.csv'), encoding="utf-8")


def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(
                 "https://thumbs.dreamstime.com/z/double-exposure-row-coins-credit-card-graph-business-finance-background-140092290.jpg");
             background-size: cover;
             background-repeat: no-repeat;
             width: 100%;
             height: 0;
             padding-top: 66.64%; /* (img-height / img-width * container-width) *//* (853 / 1280 * 100) */
             background-size: auto,
             background-size: 150px
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


set_bg_hack_url()



vtxt= "💰Credit Card Fraud Detection App💰"
htmlstr1 = f"""<p style="background-color: transparent;
    font-color: '#d60000';
    font-size: 42px;
    border-radius: 7px;
    padding-left: 12px;
    padding-top: 13px;
    padding-bottom:13px;
    line-height:25px;">
    {vtxt}</style>
    <BR></p>"""
st.markdown(htmlstr1,unsafe_allow_html=True)

# sidebar
img = Image.open("Front+cover.jpeg")
img = img.resize((250, 200))
color = '#d60000'
st.sidebar.image(img)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: url(
                 "https://i.pinimg.com/originals/29/f8/51/29f851ee50c52b29f0c2f16ffdae25f9.jpg");
             background-size: auto
}
</style>
""",
    unsafe_allow_html=True,
)

html_temp2 = """
<div style="background-color:transparent">
<h1 style="color:#d60000;text-align:center;"> Credit Card Fraud Detection</h1>
</div><br>"""
st.sidebar.markdown(html_temp2, unsafe_allow_html=True)

st.sidebar.header(
    "Credit card fraud detection is the collective term for the policies, tools, methodologies, and practices that credit card companies and financial institutions take to combat identity fraud and stop fraudulent transactions. ")
st.sidebar.header("A credit card account that doesn't require possession of a physical card. Commonly a method used to make online purchases, it requires only that the thief knows your name, account number and the card's security code.")
st.sidebar.subheader("Predict the fraud according features.")

# SON KOD
# [[ 0.        ,  3.99790559, -0.52218786, -2.53738731,  1.39165725,
#         -2.89990739, -0.59522188, -4.28925378, -2.83005567,  0.12691056,
#         -0.46521108]]

Amount = st.slider(label="Amount", min_value=0.00, max_value=23691.1,value=0.0, step=0.01) 
V4 = st.slider(label="V4", min_value=-5.00, max_value=16.0,value=3.99790559,step=0.01)
V5 = st.slider(label="V5", min_value=-113.00, max_value=34.0,value=-0.52218786, step=0.01)
V7 = st.slider(label="V7", min_value=-43.00, max_value=120.0,value=-2.53738731, step=0.01)
V8 = st.slider(label="V8", min_value=-73.00, max_value=20.0,value=1.391657251, step=0.01)
V12 = st.slider(label="V12", min_value=-18.00, max_value=7.8,value=-2.89990739, step=0.01)
V13 = st.slider(label="V13", min_value=-18.00, max_value=7.8,value=-0.59522188, step=0.01)
V14 = st.slider(label="V14", min_value=-19.00, max_value=10.0,value=-4.28925378, step=0.01)
V17 = st.slider(label="V17", min_value=-25.00, max_value=9.0,value=-2.83005567, step=0.01)
V20 = st.slider(label="V20", min_value=-54.00, max_value=39.0,value=.12691056, step=0.01)
V23 = st.slider(label="V23", min_value=-44.00, max_value=22.0,value=-0.46521108, step=0.01)

#X = df[["Amount", "V4", "V5", "V7", "V8", "V12", "V13", "V14", "V17", "V20", "V23"]]

coll_dict = {'Amount':Amount, 'V4':V4, 'V5':V5, 'V7':V7, 'V8':V8, 'V12':V12, 'V13':V13, 'V14':V14, 'V17':V17, 'V20':V20, 'V23':V23,}
columns = ["Amount", "V4", "V5", "V7", "V8", "V12", "V13", "V14", "V17", "V20", "V23"]
df_coll = pd.DataFrame.from_dict([coll_dict])



######

th_props = [
  ('font-size', '16px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'), # #6d6d6d
  ('color', 'black'),
  ('background-color', 'transparent'),("width", "850px"), # #f7ffff
            ("height","80px")
  ]
                               
td_props = [
  ('font-size', '15px')
  ]
                                 
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

df2=df_coll.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)

st.table(df2.set_properties(**{'background-color': 'lightblue',
                                        'color': '#d60000',
                                        'border-color': 'white'}))


st.subheader("Click PREDICT if configuration is OK")



myButton1 = st.button("Predict the Fraud")
button_style = """
                    <style>
                    .stButton > button {
                        color: #d60000;
                        background: transparent;
                        width: 700px;
                        height: 50px;
                        font-size: 25px;
                    }
                    </style>
                     """
st.markdown(button_style, unsafe_allow_html=True) 




if myButton1:
    prediction = model.predict(df_coll)
    if int(prediction[0])==0:
    
        st.subheader(prediction[0])

        st.subheader(f"Transaction is SAFE  :)")
    else:
        st.subheader(prediction[0])
       
        
        st.subheader(f'ALARM! Transaction is FRAUDULENT  :(')
    