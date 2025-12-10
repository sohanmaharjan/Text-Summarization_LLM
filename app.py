import streamlit as st
from transformers import pipeline

# Load the summarization model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_summarizer():
    """Loads the summarization model."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # or your saved model path
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

summarizer = load_summarizer()

st.title("Text Summarization App")

user_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if summarizer and user_input:
        try:
            summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False) # Adjust parameters as needed
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error during summarization: {e}")
    elif not user_input:
        st.warning("Please enter text to summarize.")
    else:
        st.error("Model not loaded. Please check the logs.")