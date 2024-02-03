import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

uploaded_file = st.file_uploader("ファイルをアップロード")

if uploaded_file:
    df = pd.read_csv(
        uploaded_file,
        index_col=0
    )

    st.markdown("ファイルプレビュー")
    st.write(df.head(5))

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    st.write(X)
    st.write(y)