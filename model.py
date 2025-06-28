import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.3
)
