# Task 1: Multi-Model Comparison
# Implement a function that embeds the same set of documents using at least two different models 
# (e.g., text-embedding-3-small and a HuggingFace model). 
# Compute the similarity matrix for each model and compare how they rank document pairs. Write a brief analysis of the differences.

import os
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer


openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
)


OPENAI_MODEL = "openai/text-embedding-3-small"
HF_MODEL_NAME = "all-MiniLM-L6-v2"
hf_model = SentenceTransformer(HF_MODEL_NAME)

documents = [
    "Machine learning models learn from complex patterns in data.",
    "ML algorithms are trained on massive structural datasets.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Deep learning and neural networks power modern artificial intelligence.",
    "Going for a walk in a sunny park is great for physical health."
]

def get_openrouter_embeddings(texts: list[str], model: str) -> np.ndarray:
    """Fetch embeddings from OpenRouter."""
    cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
    response = or_client.embeddings.create(
        input=cleaned_texts,
        model=model
    )
    return np.array([item.embedding for item in response.data])

def get_hf_embeddings(texts: list[str]) -> np.ndarray:
    """Fetch embeddings locally using Sentence-Transformers."""
    return hf_model.encode(texts)

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute similarity matrix for a set of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T

def display_matrix(matrix: np.ndarray, titles: list[str]):
    """Display the similarity matrix using pandas."""
    df = pd.DataFrame(matrix, columns=titles, index=titles)
    print(df.round(4))
    print("\n")

def extract_unique_pairs(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    """Extract unique pairs to avoid self-similarity and duplicates."""
    pairs = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, matrix[i, j]))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


or_embeddings = get_openrouter_embeddings(documents, OPENAI_MODEL)
hf_embeddings = get_hf_embeddings(documents)

print(f"OpenRouter ({OPENAI_MODEL}) Dimensions: {or_embeddings.shape[1]}")
print(f"HuggingFace ({HF_MODEL_NAME}) Dimensions: {hf_embeddings.shape[1]}\n")

or_sim_matrix = compute_similarity_matrix(or_embeddings)
hf_sim_matrix = compute_similarity_matrix(hf_embeddings)

short_titles = [f"Doc {i+1}" for i in range(len(documents))]

print("OpenRouter Similarity Matrix:")
display_matrix(or_sim_matrix, short_titles)

print("HuggingFace Similarity Matrix:")
display_matrix(hf_sim_matrix, short_titles)

or_pairs = extract_unique_pairs(or_sim_matrix)
hf_pairs = extract_unique_pairs(hf_sim_matrix)

print("Document Pair Ranking Comparison")
print(f"{'Rank':<5} | {'Pair':<12} | {'OpenRouter Score':<18} | {'HuggingFace Score':<18}")
print("-" * 65)

for rank, (or_p, hf_p) in enumerate(zip(or_pairs, hf_pairs), 1):
    or_label = f"Doc {or_p[0]+1} <-> {or_p[1]+1}"
    hf_label = f"Doc {hf_p[0]+1} <-> {hf_p[1]+1}"
    
    print(f"{rank:<5} | {or_label:<12} ({or_p[2]:.4f})     | {hf_label:<12} ({hf_p[2]:.4f})")