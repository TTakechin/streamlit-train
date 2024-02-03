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

    on_button = st.button("実行する")
    if on_button:
       
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg = RandomForestRegressor()
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)

        st.write("精度はf{r2_score(y_test, y_pred)}です")
        plt.figure(figsize=(8,8))
        plt.scatter(y_train, reg.predict(X_train), label="Train")
        plt.scatter(y_test, reg.predict(X_test), label="Test")
        plt.plot(np.arange(0,60),np.arange(0,60),linestyle="dashed", color="gray")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.legend()