# Task 1: Implement Positional Encoding and Integrate with Attention
# Write a function that generates sinusoidal positional encodings for any sequence length and model dimension.
# Create a toy input matrix of shape (6, 8) representing 6 tokens with 8-dimensional embeddings.
# Add the positional encoding to the input, then run self-attention on both the un-encoded and encoded inputs.
# Print the attention weight matrices side by side and explain in 3-5 sentences how positional encoding changes the attention patterns.

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Queries  - shape (seq_len, d_k) - "what am I looking for?"
        K: Keys     - shape (seq_len, d_k) - "what do I contain?"
        V: Values   - shape (seq_len, d_v) - "what info do I provide?"
        mask: Optional mask - shape (seq_len, seq_len)

    Returns:
        output:            Contextualized representations (seq_len, d_v)
        attention_weights: How much each token attends to others (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute dot product of queries and keys
    scores = Q @ K.T                          # (seq_len, seq_len)

    # Step 2: Scale by 1/sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (for causal/decoder attention or padding)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax to get attention weights
    attention_weights = softmax(scores)       # (seq_len, seq_len)

    # Step 5: Multiply weights by values
    output = attention_weights @ V            # (seq_len, d_v)

    return output, attention_weights


def positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding from the Transformer paper.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)

    PE[:, 0::2] = np.sin(position / div_term)  # Even indices
    PE[:, 1::2] = np.cos(position / div_term)  # Odd indices

    return PE


def main():

    
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    d_model = 8
    d_k = 4
    d_v = 4
    
    np.random.seed(42)

    X = np.random.randn(seq_len, d_model) * 0.5
    X[4] = X[0] + np.random.randn(d_model) * 0.05

    # Learned projection matrices
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_v) * 0.1

    # Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    Q_no_pe = X @ W_Q
    K_no_pe = X @ W_K
    V_no_pe = X @ W_V
    _, attn_no_pe = scaled_dot_product_attention(Q_no_pe, K_no_pe, V_no_pe)
    
    PE = positional_encoding(seq_len, d_model)
    X_with_pe = X + PE
    
    Q_pe = X_with_pe @ W_Q
    K_pe = X_with_pe @ W_K
    V_pe = X_with_pe @ W_V
    _, attn_pe = scaled_dot_product_attention(Q_pe, K_pe, V_pe)
    

    print("=" * 60)
    print("COMPARING ATTENTION MAPS: UN-ENCODED VS. POSITIONAL ENCODED INPUTS")
    print("=" * 60)

    print(f"{'WITHOUT POSITIONAL ENCODING':<36} | {'WITH POSITIONAL ENCODING':<36}")
    print(f"{'':>6}", end="")
    for t in tokens: print(f"{t:>5}", end="")
    print(" | ", end="")
    for t in tokens: print(f"{t:>5}", end="")
    print("\n" + "-" * 60)
    

    for i in range(seq_len):
        print(f"{tokens[i]:>5} ", end="")
        for j in range(seq_len):
            print(f"{attn_no_pe[i, j]:.2f} ", end="")
        
        print(" | ", end="")
        
        for j in range(seq_len):
            print(f"{attn_pe[i, j]:.2f} ", end="")
        print()
    print("-" * 60)

if __name__ == "__main__":
    main()

# Without positional encodings, the model is completely blind to where anyone is sitting. 
# It doesn't know who is at the head of the table or who is sitting next to whom; it only knows what people are saying.
# The attention matrix on the left shows this: the connections are based purely on how similar the words are, ignoring the physical gaps between them.

# When we add Positional Encodings, we are essentially handing every person a unique name tag 
# that says exactly what number chair they are sitting in (Chair 0, Chair 1, Chair 2, etc.).
# Instead of just looking at the word itself, the model now looks at Word Meaning + Chair Number.