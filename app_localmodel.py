import streamlit as st
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the summarization model using pickle
@st.cache_resource
def load_pickled_summarizer(model_path):
    """Loads a pickled summarization model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model from pickle: {e}")
        return None

model_path = 'model.pkl'
summarizer = load_pickled_summarizer(model_path)
print(summarizer)

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = summarizer.to(device)

def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(**inputs, max_length=128)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.title("Text Summarization App")

user_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if summarizer and user_input:
        try:
            # Assuming your pickled model has a 'summarize' method, adjust as needed.
            summary = generate_summary(user_input)
            st.subheader("Summary:")
            st.write(summary)
        except AttributeError:
            st.error("Model does not have a 'summarize' method. Please check your model.")
        except Exception as e:
            st.error(f"Error during summarization: {e}")
    elif not user_input:
        st.warning("Please enter text to summarize.")
    else:
        st.error("Model not loaded. Please check the logs and model path.")