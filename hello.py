

import numpy as np
import pandas as pd
import streamlit as st
import pickle


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="NYC Ridesharing Demo", page_icon=":chart_with_upwards_trend:")

st.title("Credit Card Approval ")

df = pd.read_csv('cleaned_df')

model = pickle.load(open('credit_model.sav','rb'))

sample = df.sample()

yy = sample.drop(['high_risk','Unnamed: 0'],axis=1)

yy

result = model.predict(yy)

result[0]

st.write(
"""
## Gender

""")

input_age = np.negative(st.slider('Select your age', value=42, min_value=18, max_value=70,step=1) *365.25)

age = np.abs(input_age)
age