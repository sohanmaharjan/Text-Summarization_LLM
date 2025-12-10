# Text Summarization using LLM (Samsum Dataset)

This project demonstrates a **text summarization model** trained on the **Samsum dataset** using a **Large Language Model (LLM)**. The model generates concise summaries of conversations and has been integrated into a simple **Streamlit web app** for local testing and demonstration.

---

## Objective

The purpose of this project is to:
- Fine-tune a pre-trained LLM for abstractive summarization
- Learn and apply text preprocessing and dataset preparation techniques
- Deploy the trained model in a local web app using Streamlit
- Practice model saving/loading and web interface development

---

## What I Learned

- How to train and evaluate a text summarization model on dialogue data  
- Hands-on experience with the **Samsum dataset**, ideal for conversational text  
- Model serialization using **Pickle**  
- Creating a simple and interactive **Streamlit** app for text input and summary generation  
- Running and testing models locally for rapid development and iteration

---

## Tools & Technologies

- **Python**
- **Hugging Face Transformers**  
- **Samsum Dataset**  
- **Streamlit**
- **Pickle**
- **Jupyter Notebook**

---

## Project Structure

- `train_model.ipynb` – Notebook for preprocessing and training the summarization model  
- `app.py` – Python script for the Streamlit web app  
- `samsum_dataset.csv` – Source dataset (or loaded via Hugging Face)  
- `model.pkl` – Saved model file using Pickle  
- `requirements.txt` – Python dependencies for the project

---

How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-summarization-llm.git
   cd text-summarization-llm
