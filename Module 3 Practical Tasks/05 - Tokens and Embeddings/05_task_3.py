# Task 3: Embedding Similarity Analysis

# Using the OpenAI Embeddings API (or a pre-trained word2vec model from gensim), generate embeddings for 20 words across 4 semantic categories 
# (e.g., animals, vehicles, emotions, professions). Compute the cosine similarity matrix between all 20 words. 
# Create a visualization (text-based heatmap is acceptable) showing the similarity matrix. 
# Calculate the average within-category similarity and between-category similarity. Write a short analysis (5-7 sentences) discussing 
# whether the embeddings correctly cluster semantically similar words and any surprising results you found.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

categories = {
    "Animals": ["lion", "tiger", "elephant", "wolf", "deer"],
    "Vehicles": ["car", "motorcycle", "airplane", "train", "bicycle"],
    "Emotions": ["happiness", "sadness", "anger", "fear", "excitement"],
    "Professions": ["doctor", "engineer", "teacher", "lawyer", "artist"]
}

words = []
word_to_category = {}
for cat, word_list in categories.items():
    words.extend(word_list)
    for word in word_list:
        word_to_category[word] = cat

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(words)

similarity_matrix = cosine_similarity(embeddings)

within_similarities = []
between_similarities = []

for i in range(len(words)):
    for j in range(i + 1, len(words)):
        sim_val = similarity_matrix[i][j]
        if word_to_category[words[i]] == word_to_category[words[j]]:
            within_similarities.append(sim_val)
        else:
            between_similarities.append(sim_val)

avg_within = np.mean(within_similarities)
avg_between = np.mean(between_similarities)

print(f"Average Within-Category Similarity: {avg_within:.4f}")
print(f"Average Between-Category Similarity: {avg_between:.4f}\n")

plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix, 
    xticklabels=words, 
    yticklabels=words, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu",
    cbar_kws={'label': 'Cosine Similarity'}
)
plt.title("Embedding Cosine Similarity Matrix Heatmap", fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()