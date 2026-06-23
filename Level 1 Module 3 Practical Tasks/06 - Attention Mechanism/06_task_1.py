# Task 1: Implement and Visualize Attention for a Real Sentence
# Choose a sentence with at least 8 words that contains a pronoun reference (e.g., "The programmer told the designer that she needed to fix the bug"). 
# Create manual embeddings by assigning each word a 6-dimensional vector where you deliberately encode semantic relationships 
# (similar words get similar vectors). Implement scaled dot-product attention and compute the attention matrix. 
# Visualize the attention weights as a formatted table and write 5-7 sentences analyzing which words the pronoun ("she") attends to most and why.

import numpy as np

tokens = ["The", "programmer", "told", "the", "designer", "that", "she", "needed", "to", "fix", "the", "bug"]
seq_len = len(tokens)
d_model = 6
d_k = 6
d_v = 6

X = np.array([
    [0.1, 0.0, 0.0, 0.0, 0.9, 0.1],  # The
    [0.9, 0.8, 0.1, 0.0, 0.1, 0.2],  # programmer
    [0.2, 0.0, 0.0, 0.8, 0.2, 0.5],  # told
    [0.1, 0.0, 0.0, 0.0, 0.9, 0.1],  # the
    [0.9, 0.1, 0.8, 0.0, 0.1, 0.2],  # designer
    [0.1, 0.0, 0.0, 0.0, 0.8, 0.6],  # that
    [0.9, 0.75, 0.15, 0.0, 0.4, 0.1], # she
    [0.3, 0.0, 0.0, 0.7, 0.2, 0.4],  # needed
    [0.1, 0.0, 0.0, 0.1, 0.8, 0.2],  # to
    [0.4, 0.4, 0.2, 0.8, 0.1, 0.3],  # fix
    [0.1, 0.0, 0.0, 0.0, 0.9, 0.1],  # the
    [0.1, 0.5, 0.0, 0.0, 0.2, 0.8],  # bug
])

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

W_Q = np.eye(d_model, d_k)
W_K = np.eye(d_model, d_k)
W_V = np.eye(d_model, d_v)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T
scaled_scores = scores / np.sqrt(d_k)
attention_weights = softmax(scaled_scores)

#print(attention_weights @ V)

print(f"\n{'':<12}", end="")
for t in tokens:
    print(f"{t:>11}", end="")
print("\n" + "-" * 145)

for i, token in enumerate(tokens):
    print(f"{token:<12}", end="")
    for j in range(seq_len):
        print(f"{attention_weights[i, j]:11.3f}", end="")
    print()