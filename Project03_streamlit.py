import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import time

import pandas as pd
import numpy as np

import xgboost as xgb
import zipfile
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection App", page_icon="💰",
                   layout='centered', initial_sidebar_state='expanded')

zf = zipfile.ZipFile('./creditcard.zip') 
# if you want to see all files inside zip folder
#print(zf.namelist())
df = pd.read_csv(zf.open('creditcard.csv'), encoding="utf-8")

model = pickle.load(open('Fraud_Detection_xgb.pkl', 'rb'))


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
         """, unsafe_allow_html=True)
set_bg_hack_url()

vtxt= "💰Credit Card Fraud Detection App💰"
htmlstr1 = f"""<p style="background-color: transparent;
    font-color: '#d60000';
    font-size: 42px;
    border-radius: 7px;
    padding-left: 12px;
    padding-top: 13px;
    padding-bottom:13px;
    line-height:25px;">{vtxt}</style><br></p>"""
st.markdown(htmlstr1, unsafe_allow_html=True)



# Creating side bar 
# st.sidebar.header("User input parameter")
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
    """, unsafe_allow_html=True)

html_temp2 = """
<div style="background-color:transparent">
<h1 style="color:#d60000;text-align:center;"> Credit Card Fraud Detection</h1>
</div><br>"""
st.sidebar.markdown(html_temp2, unsafe_allow_html=True)
st.sidebar.header(
    "Credit card fraud detection is the collective term for the policies, tools, methodologies, and practices that credit card companies and financial institutions take to combat identity fraud and stop fraudulent transactions. ")
st.sidebar.header("A credit card account that doesn't require possession of a physical card. Commonly a method used to make online purchases, it requires only that the thief knows your name, account number and the card's security code.")
st.sidebar.subheader("Predict the fraud according features.")
html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">   </h1>
</div><br>"""
st.sidebar.markdown(html_temp,unsafe_allow_html=True)



# SON KOD
# [[ 0.        ,  3.99790559, -0.52218786, -2.53738731,  1.39165725,
#         -2.89990739, -0.59522188, -4.28925378, -2.83005567,  0.12691056,
#         -0.46521108]]


# def user_input_data():
#     Amount = st.slider(label="Amount", min_value=min(df['Amount']), max_value=max(df['Amount']), value=0.0, step=0.01) 
#     V4 = st.slider(label="V4", min_value=min(df['V4']), max_value=max(df['V4']), value=3.99790559,step=0.01)
#     V5 = st.slider(label="V5", min_value=min(df['V5']), max_value=max(df['V5']), value=-0.52218786, step=0.01)
#     V7 = st.slider(label="V7", min_value=min(df['V7']), max_value=max(df['V7']), value=-2.53738731, step=0.01)
#     V8 = st.slider(label="V8", min_value=min(df['V8']), max_value=max(df['V8']), value=1.391657251, step=0.01)
#     V12 = st.slider(label="V12", min_value=min(df['V12']), max_value=max(df['V12']), value=-2.89990739, step=0.01)
#     V13 = st.slider(label="V13", min_value=min(df['V13']), max_value=max(df['V13']), value=-0.59522188, step=0.01)
#     V14 = st.slider(label="V14", min_value=min(df['V14']), max_value=max(df['V14']), value=-4.28925378, step=0.01)
#     V17 = st.slider(label="V17", min_value=min(df['V17']), max_value=max(df['V17']), value=-2.83005567, step=0.01)
#     V20 = st.slider(label="V20", min_value=min(df['V20']), max_value=max(df['V20']), value=.12691056, step=0.01)
#     V23 = st.slider(label="V23", min_value=min(df['V23']), max_value=max(df['V23']), value=-0.46521108, step=0.01)    

def user_input_data():
    Amount = st.slider(label="Amount", min_value=min(df['Amount']), max_value=max(df['Amount']), step=0.01) 
    V4 = st.slider(label="V4", min_value=min(df['V4']), max_value=max(df['V4']),step=0.01)
    V5 = st.slider(label="V5", min_value=min(df['V5']), max_value=max(df['V5']), step=0.01)
    V7 = st.slider(label="V7", min_value=min(df['V7']), max_value=max(df['V7']), step=0.01)
    V8 = st.slider(label="V8", min_value=min(df['V8']), max_value=max(df['V8']), step=0.01)
    V12 = st.slider(label="V12", min_value=min(df['V12']), max_value=max(df['V12']), step=0.01)
    V13 = st.slider(label="V13", min_value=min(df['V13']), max_value=max(df['V13']), step=0.01)
    V14 = st.slider(label="V14", min_value=min(df['V14']), max_value=max(df['V14']), step=0.01)
    V17 = st.slider(label="V17", min_value=min(df['V17']), max_value=max(df['V17']), step=0.01)
    V20 = st.slider(label="V20", min_value=min(df['V20']), max_value=max(df['V20']), step=0.01)
    V23 = st.slider(label="V23", min_value=min(df['V23']), max_value=max(df['V23']), step=0.01)    
 
    data = { 
        'Amount': Amount,
        'V4': V4,
        'V5': V5,
        'V7': V7,
        'V8': V8,
        'V12': V12,
        'V13': V13,
        'V14': V14,
        'V17': V17,
        'V20': V20,
        'V23': V23,
    }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data


df = user_input_data() 


if st.checkbox('Show User Inputs:', value=True):
#         st.write(df.astype(str).rename(columns={0:'input_data'}))
    th_props = [
      ('font-size', '16px'),
      ('text-align', 'center'),
      ('font-weight', 'bold'), # #6d6d6d
      ('color', 'dodgerblue'),
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

    df2=df.style.set_properties(**{'text-align': 'left', 'fontweight': 'bold'}).set_table_styles(styles)

    st.table(df2.set_properties(**{'background-color': 'lightblue',
                                            'color': '#d60000',
                                            'border-color': 'white'}))


if st.button('Make Prediction'):        
    prediction = model.predict(df)
    
    if int(prediction[0])==0:    
        st.success(prediction[0])
        st.success(f"Transaction is SAFE  :)")
        
    else:
        st.warning(prediction[0])        
        st.warning(f'ALARM! Transaction is FRAUDULENT  :(')
