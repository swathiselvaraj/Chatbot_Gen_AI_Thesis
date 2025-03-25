import streamlit as st

st.title("Hello, I'm your AI Chatbot!")
user_input = st.text_input("Ask me something:")

if user_input:
    st.write(f"You asked: {user_input}")