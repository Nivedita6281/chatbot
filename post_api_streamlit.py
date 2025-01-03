import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000/qa"

# Streamlit UI
st.title("Question Answering System")
st.write("Ask any question based on the documents loaded in the backend.")

# Input for the question
question = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Fetching answer..."):
            try:
                # Send the question to the FastAPI backend
                response = requests.post(BACKEND_URL, json={"question": question})
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided.")
                    st.success("Answer:")
                    st.write(answer)
                else:
                    st.error("Error: Unable to fetch the answer. Please check the backend.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid question.")
