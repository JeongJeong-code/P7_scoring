# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 23:40:07 2022

@author: nwenz
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.title("test_streamlit")

path =r'C:\Users\nwenz\Desktop\P7_scoring/'


@st.cache
def data_load():
    data = pd.read_csv(path + 'rfc_results.csv')
    #df_test = pd.read_csv(r'C:\Users\nwenz\Desktop\P7_scoring\Projet+Mise+en+prod+-+home-credit-default-risk\\' + r'application_test.csv')
    return(data)
df_test = data_load()
#st.dataframe(df_test)
with st.sidebar:
    client = st.selectbox("client ID",df_test.SK_ID_CURR)
    categories = st.multiselect("categories display",df_test.columns)
    prediction = st.button(label ='prédire')


st.table(df_test[df_test['SK_ID_CURR'] == client])

for cat in categories :

        fig, ax = plt.subplots()
        ax.bar(df_test[cat],height =10)
        st.pyplot(fig)


