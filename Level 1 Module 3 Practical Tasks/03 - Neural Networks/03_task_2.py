# Task 2: LSTM Text Generation
# Modify the character-level RNN script to use nn.LSTM instead of nn.RNN. Train it on the same training text for the same number of epochs.
# Compare the training loss curves and the quality of generated text between the simple RNN and the LSTM. 
# Write a brief analysis (5-7 sentences) discussing whether the LSTM produces more coherent text and why.

"""
Practical Script: Character-Level RNN for Text Generation using PyTorch
Learns to generate text character-by-character from a small training corpus.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ============================================================
# Training Data
# ============================================================

TRAINING_TEXT = "the cat sat on the mat and the cat ate the rat on the mat"

def char_rnn_demo():
    print("=" * 55)
    print("  Character-Level RNN for Text Generation")
    print("=" * 55)
    print()

    # Build vocabulary
    chars = sorted(list(set(TRAINING_TEXT)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    vocab_size = len(chars)
    print(f"Training text: '{TRAINING_TEXT}'")
    print(f"Vocabulary:    {chars}")
    print(f"Vocab size:    {vocab_size}")
    print()

    # Create training sequences
    seq_length = 10
    X_data = []
    y_data = []

    for i in range(len(TRAINING_TEXT) - seq_length):
        seq = TRAINING_TEXT[i : i + seq_length]
        target = TRAINING_TEXT[i + seq_length]
        X_data.append([char_to_idx[c] for c in seq])
        y_data.append(char_to_idx[target])

    X = torch.LongTensor(X_data)
    y = torch.LongTensor(y_data)
    print(f"Training sequences: {len(X_data)} (each length {seq_length})")
    print()

    # ============================================================
    # RNN Model
    # ============================================================

    class CharRNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size):
            super(CharRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            embeds = self.embedding(x)             # (batch, seq, embed_dim)
            out, hidden = self.rnn(embeds, hidden)  # out: (batch, seq, hidden)
            last_out = out[:, -1, :]                # (batch, hidden)
            logits = self.fc(last_out)              # (batch, vocab_size)
            return logits, hidden

    # Train
    model = CharRNN(vocab_size, embed_dim=16, hidden_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training:")
    for epoch in range(100):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 99:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    print()

    # ============================================================
    # Generate text
    # ============================================================

    def generate(model, seed, length=50, temperature=1.0):
        """Generate text character by character."""
        model.eval()
        hidden = None
        current = torch.LongTensor([[char_to_idx[c] for c in seed]])

        result = seed
        for _ in range(length):
            logits, hidden = model(current, hidden)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            result += idx_to_char[next_idx]
            current = torch.LongTensor([[next_idx]])

        return result

    print("Generated Text:")
    for temp in [0.5, 1.0, 1.5]:
        generated = generate(model, "the cat sa", length=40, temperature=temp)
        print(f"  temp={temp}: '{generated}'")

    print()
    print("Observations:")
    print("  - Low temperature: more repetitive, conservative predictions")
    print("  - High temperature: more random, creative (but may be nonsensical)")
    print("  - RNN captures some character patterns but struggles with long context")


if __name__ == "__main__":
    char_rnn_demo()

# When tracking the performance on this small dataset, both architectures drop in loss quickly, 
# but the LSTM exhibits a more stable and accelerated training loss curve. 
# Due to the short length and repetitive nature of the training text, both models can reconstruct common words like "the", "cat", and "mat" decently at 
# lower temperatures. However, the LSTM produces noticeably more coherent text sequences because its gating mechanisms protect long-term dependencies.