import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

# Model and tokenizer loading
checkpoint = "google/flan-t5-small"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float32, 
    device_map="auto" if torch.cuda.is_available() else None
)
base_model.to(device)

# Load the retrieval model (Sentence-BERT) and FAISS index
retriever_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
corpus_embeddings = np.load(r"C:\Users\piyus\Documents\LaMini-LM-Summarization-Streamlit-App-main\corpus_embeddings.npy")
 # Load precomputed embeddings
corpus_texts = [
    "Document 1 content here.",
    "Document 2 content here.",
    "Document 3 content here."
]  # Replace with your actual text documents

# Ensure the number of documents matches the embeddings
assert len(corpus_texts) == corpus_embeddings.shape[0], "Mismatch between embeddings and corpus texts."

faiss_index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
faiss_index.add(corpus_embeddings)

# Function to preprocess PDF file
def file_preprocessing(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()

# Retrieval function
def retrieve_documents(query_text, k=5):
    query_embedding = retriever_model.encode([query_text])
    D, I = faiss_index.search(query_embedding, k)  # Retrieve top-k documents
    retrieved_texts = [corpus_texts[idx] for idx in I[0] if idx < len(corpus_texts)]  # Safeguard against out-of-bound indices
    return " ".join(retrieved_texts)

# LLM pipeline with retrieval-augmented generation (RAG)
def llm_pipeline(file_path):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # Auto-detect GPU if available
        max_length=500,
        min_length=50
    )
    
    input_text = file_preprocessing(file_path)
    retrieved_context = retrieve_documents(input_text)  # Retrieve relevant documents
    augmented_input = retrieved_context + " " + input_text  # Combine retrieved text with original input
    
    result = pipe_sum(augmented_input)
    return result[0]['summary_text']

# Function to display the PDF
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using RAG")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            file_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)  # Ensure the directory exists

            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                displayPDF(file_path)

            with col2:
                try:
                    summary = llm_pipeline(file_path)
                    st.info("Summarization Complete")
                    st.success(summary)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
