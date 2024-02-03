# https://www.hello-statisticians.com/python/data-viz-streamlit-text-01.html

import streamlit as st
import test

name1 = test.main()
st.title("Quiz")
st.title(name1)
st.markdown("""
### 日本の首都を選択肢より選びなさい。
"""
)

if st.checkbox("札幌"):
    st.markdown("残念、不正解...(T_T)")
if st.checkbox("東京"):
    st.markdown("正解！")
if st.checkbox("大阪"):
    st.markdown("残念、不正解... (T_T)")

