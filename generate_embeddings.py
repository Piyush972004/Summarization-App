from sentence_transformers import SentenceTransformer
import numpy as np

# Define your corpus texts (replace with your actual documents)
corpus_texts = [
    "Document 1 content goes here.",
    "Document 2 content goes here.",
    "Document 3 content goes here.",
    # Add more documents here
]

# Load the Sentence-BERT model
retriever_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for the corpus
corpus_embeddings = retriever_model.encode(corpus_texts)

# Save the embeddings to a .npy file
np.save("corpus_embeddings.npy", corpus_embeddings)

print("Embeddings saved successfully to corpus_embeddings.npy")