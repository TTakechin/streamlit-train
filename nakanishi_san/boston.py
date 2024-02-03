import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

uploaded_file = st.file_uploader("boston.csvをアップロード")
model = pickle.load(open("train.pkl","rb"))
if uploaded_file:
    df = pd.read_csv(uploaded_file,index_col=0)
    st.markdown("ファイルプレビュー")
    st.write(df.head(5))

    X= df.drop("medv",axis=1)
    y = df["medv"]

    y_pred = model.predict(X)

    st.write(f"精度は{r2_score(y, y_pred)}です")
    

