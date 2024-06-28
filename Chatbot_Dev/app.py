import streamlit as st
from model.bert_model import get_response

# Title of the app
st.title("Cardio-Guardian ChatBot")

# Add a logo (if needed)
st.image("static/logo.png", use_column_width=True)

# User input
user_input = st.text_input("Ask a question about heart disease:")

if user_input:
    # Get the response from the BERT model
    response = get_response(user_input)
    # Display the response
    st.write("Bot:", response)
