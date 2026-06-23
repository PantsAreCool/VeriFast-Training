# Task 2: Compare Self-Attention, Cross-Attention, and Causal Attention
# Implement all three attention types in a single script. For self-attention, process "The weather is beautiful today". 
# For cross-attention, use English tokens "The cat" as the source and French tokens "Le chat" as the query. 
# For causal attention, process "I am learning about attention" with masking. Print the attention matrices for all three side by side. 
# Write a comparison (5-7 sentences) explaining how the attention patterns differ and when each type would be used.

import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

def print_matrix(row_tokens, col_tokens, weights, title):
    print(f"\n--- {title} ---")
    print(f"{'':<12}", end="")
    for t in col_tokens:
        print(f"{t:>12}", end="")
    print("\n" + "-" * (12 + 12 * len(col_tokens)))
    for i, r_tok in enumerate(row_tokens):
        print(f"{r_tok:<12}", end="")
        for j in range(len(col_tokens)):
            print(f"{weights[i, j]:12.3f}", end="")
        print()

np.random.seed(101)
d_model, d_k, d_v = 8, 8, 8
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1



# 1. Self-Attention

tokens_self = ["The", "weather", "is", "beautiful", "today"]
X_self = np.random.randn(len(tokens_self), d_model)
Q_s, K_s, V_s = X_self @ W_Q, X_self @ W_K, X_self @ W_V
_, weights_self = scaled_dot_product_attention(Q_s, K_s, V_s)



# 2. Cross-Attention

tokens_en_source = ["The", "cat"]
tokens_fr_query = ["Le", "chat"]
X_en = np.random.randn(len(tokens_en_source), d_model) # Source sequence
X_fr = np.random.randn(len(tokens_fr_query), d_model)   # Query sequence
Q_c = X_fr @ W_Q
K_c, V_c = X_en @ W_K, X_en @ W_V
_, weights_cross = scaled_dot_product_attention(Q_c, K_c, V_c)



# 3. Causal (Masked) Attention

tokens_causal = ["I", "am", "learning", "about", "attention"]
X_causal = np.random.randn(len(tokens_causal), d_model)
Q_m, K_m, V_m = X_causal @ W_Q, X_causal @ W_K, X_causal @ W_V
mask = np.tril(np.ones((len(tokens_causal), len(tokens_causal))))
_, weights_causal = scaled_dot_product_attention(Q_m, K_m, V_m, mask=mask)



print_matrix(tokens_self, tokens_self, weights_self, "Self-Attention Weights")
print_matrix(tokens_fr_query, tokens_en_source, weights_cross, "Cross-Attention Weights (FR -> EN)")
print_matrix(tokens_causal, tokens_causal, weights_causal, "Causal Masked Attention Weights")