# Task 2: Embedding Dimensionality Reduction
# Using the dimensions parameter of text-embedding-3-large, generate embeddings at 256, 512, 1024, 2048, and 3072 dimensions. 
# For each dimensionality, compute search accuracy on a set of 20 documents with known relevant results. 
# Plot dimensionality vs. accuracy to find the sweet spot.

import os
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
)
MODEL_NAME = "openai/text-embedding-3-large"


documents = [
    "The planet Mars has a thin atmosphere and a cold, desert-like surface.", # Doc 0
    "Quantum computing utilizes qubits to perform complex calculations rapidly.", # Doc 1
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.", # Doc 2
    "The Federal Reserve adjust interest rates to manage inflation and growth.", # Doc 3
    "Renaissance art was characterized by realism, perspective, and humanism.", # Doc 4
    "Python uses dynamic typing and automatic memory management via GC.", # Doc 5
    "DNA replication ensures that genetic information is accurately copied.", # Doc 6
    "The Treaty of Versailles formally ended World War I in 1919.", # Doc 7
    "Deep neural networks consist of an input layer, hidden layers, and output.", # Doc 8
    "Macroeconomics studies economy-wide phenomena like GDP and unemployment.", # Doc 9
    "Impressionist painters like Monet focused on light and movement.", # Doc 10
    "Black holes possess an event horizon from which nothing can escape.", # Doc 11
    "Mitosis is a process of cell division resulting in two identical cells.", # Doc 12
    "Scrum is an agile framework that uses sprints and daily standups.", # Doc 13
    "The Great Depression was a severe worldwide economic depression in the 1930s.", # Doc 14
    "Mitochondria are the powerhouses of the cell, generating ATP energy.", # Doc 15
    "A vector database allows for fast semantic searching using embeddings.", # Doc 16
    "The French Revolution led to the rise of Napoleon Bonaparte.", # Doc 17
    "Docker containers package applications with all their dependencies.", # Doc 18
    "Cryptographic hash functions turn arbitrary data into fixed-size strings." # Doc 19
]


eval_dataset = [
    ("Tell me about cells dividing and reproducing.", 12),
    ("How does the central bank control inflation?", 3),
    ("What are the characteristics of painting in the Renaissance?", 4),
    ("What happens inside a black hole in space?", 11),
    ("How do applications use embeddings for fast search?", 16)
]

def get_embeddings(texts: list[str], dims: int) -> np.ndarray:
    """Fetch embeddings truncated to specific dimensions."""
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    response = client.embeddings.create(
        input=cleaned,
        model=MODEL_NAME,
        dimensions=dims
    )
    return np.array([item.embedding for item in response.data])

def evaluate_accuracy(doc_embeddings: np.ndarray, queries: list[tuple[str, int]], dims: int) -> float:
    """Calculate accuracy: Is the correct document ranked #1?"""
    query_texts = [q[0] for q in queries]
    query_embeddings = get_embeddings(query_texts, dims)
    
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    normalized_docs = doc_embeddings / doc_norms
    
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized_queries = query_embeddings / query_norms
    
    similarity_matrix = normalized_queries @ normalized_docs.T
    
    hits = 0
    for idx, (_, ground_truth_doc_idx) in enumerate(queries):
        predicted_idx = np.argmax(similarity_matrix[idx])
        if predicted_idx == ground_truth_doc_idx:
            hits += 1
            
    return hits / len(queries)

dimensions_to_test = [256, 512, 1024, 2048, 3072]
accuracy_results = []

print("Starting Dimensionality Reduction Evaluation...")
for dims in dimensions_to_test:
    print(f"  Embedding documents at {dims} dimensions...")
    doc_embeddings = get_embeddings(documents, dims)
    
    accuracy = evaluate_accuracy(doc_embeddings, eval_dataset, dims)
    accuracy_results.append(accuracy)
    print(f"  -> Search Accuracy at {dims} dims: {accuracy * 100:.1f}%")

plt.figure(figsize=(10, 6))
plt.plot(dimensions_to_test, accuracy_results, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('Embedding Dimensionality vs. Search Accuracy', fontsize=14)
plt.xlabel('Dimensions', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xticks(dimensions_to_test)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)

for d, acc in zip(dimensions_to_test, accuracy_results):
    plt.annotate(f"{acc*100:.0f}%", (d, acc), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()