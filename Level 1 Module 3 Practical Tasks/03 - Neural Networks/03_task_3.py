# Task 3: Sequence Classification with RNNs
# Create a synthetic dataset of binary sequences (e.g., sequences of 0s and 1s) where sequences containing the pattern "111" are labeled class 1 and 
# all others are class 0. Build an RNN classifier using PyTorch that reads the sequence and predicts the class. 
# Train it on 500 sequences of length 20 and evaluate on 100 test sequences. 
# Report accuracy and discuss why detecting patterns in sequences is challenging for RNNs.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# Generate Training Data
# ============================================================

def generate_sequence_dataset(n_samples=600, seq_length=20):
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        seq = np.random.randint(0, 2, seq_length)
        
        label = 0
        for i in range(seq_length - 2):
            if seq[i] == 1 and seq[i+1] == 1 and seq[i+2] == 1:
                label = 1
                break
                
        X_data.append(seq)
        y_data.append(label)
        
    X = torch.FloatTensor(np.array(X_data)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y_data)).unsqueeze(-1)
    return X, y

# ============================================================
# RNN Model
# ============================================================

class SequenceClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(SequenceClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return self.sigmoid(logits)




if __name__ == "__main__":
    X, y = generate_sequence_dataset(n_samples=600, seq_length=20)
    X_train, X_test = X[:500], X[500:]
    y_train, y_test = y[:500], y[500:]

    model = SequenceClassifier(input_dim=1, hidden_dim=16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training Loop
    print("Training Sequence Classifier...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {loss.item():.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        predicted_classes = (test_preds >= 0.5).float()
        accuracy = (predicted_classes == y_test).float().mean()
        
    print(f"\nFinal Test Accuracy: {accuracy.item():.2%}")