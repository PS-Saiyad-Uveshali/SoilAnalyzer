import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.3
)

