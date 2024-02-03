import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    def show_heatmap(df):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    show_heatmap(df)
    
    column_names = df.columns
    target_column = st.selectbox("目的変数をを選んでください",column_names)
    columns_to_exclude = [target_column]
    filtered_columns = [col for col in df.columns if col not in columns_to_exclude]
    feature_select = st.radio("説明変数はどうしますか？",("目的変数以外すべて","自分で選ぶ"))
    if feature_select == "目的変数以外すべて":
        feature_list = filtered_columns
    else:
        feature_list = st.multiselect("説明変数を選んでください",filtered_columns)

    on_button = st.button("実行する")
    if on_button:
       
        X = df[feature_list]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg = RandomForestRegressor()
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)

        st.write(f"精度は{r2_score(y_test, y_pred)}です")
        fig = plt.figure()
        plt.scatter(y_train, reg.predict(X_train), label="Train")
        plt.scatter(y_test, reg.predict(X_test), label="Test")
        plt.plot(np.arange(0,60),np.arange(0,60),linestyle="dashed", color="gray")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.legend()

        st.pyplot(fig)