import streamlit as st
import pandas as pd
from chatbot_utils import preprocess, get_response

st.title("College Admission Chatbot")

data = pd.read_csv("intents.csv")

user_query = st.text_input("Ask your admission query")

if st.button("Submit"):
    if user_query:
        response = get_response(user_query, data)
        st.write("Chatbot:", response)
