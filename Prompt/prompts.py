# pyrefly: ignore [missing-import]
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header("research")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=100,
    temperature=1,
)

model = ChatHuggingFace(llm=llm)

user_input=st.text_input("enter input")

if st.button("button"):
    result=model.invoke(user_input)
    st.write(result.content)
