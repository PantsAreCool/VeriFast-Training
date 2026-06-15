# Task 1: Experiment with Network Architecture
## Modify the practical script to change the hidden layer size. Try hidden sizes of 2, 4, 8, and 16. 
## For each, train the network on XOR and record the number of epochs needed to reach 100% accuracy 
## (or the final accuracy after 2000 epochs). Write a brief analysis of how hidden layer size affects learning speed and reliability.

"""
Practical Script: 2-Layer Neural Network from Scratch using only NumPy
Solves the XOR problem -- a classic demonstration that multi-layer
networks can learn nonlinear decision boundaries.

Architecture:
  Input (2) --> Hidden Layer (4, tanh) --> Output (1, sigmoid)

Training: manual forward pass, loss computation, backpropagation, gradient update
"""

import numpy as np


# ============================================================
# Activation functions and their derivatives
# ============================================================

def sigmoid(z):
    """Sigmoid activation: output in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Derivative of sigmoid given the activated output a = sigmoid(z)."""
    return a * (1 - a)

def tanh_fn(z):
    """Tanh activation: output in (-1, 1)."""
    return np.tanh(z)

def tanh_derivative(a):
    """Derivative of tanh given the activated output a = tanh(z)."""
    return 1 - a ** 2


# ============================================================
# Loss function
# ============================================================

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================
# Neural Network class
# ============================================================

class NeuralNetwork:
    """
    A 2-layer neural network for binary classification.

    Architecture: Input(2) -> Hidden(4) -> Output(1)
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        # Initialize weights with Xavier initialization
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros((1, output_size))

        # Store for gradients
        self.grads = {} 

    def forward(self, X):
        """Forward pass through the network."""
        # Layer 1: Input -> Hidden
        self.z1 = X @ self.W1 + self.b1       # (N, hidden_size)
        self.a1 = tanh_fn(self.z1)             # (N, hidden_size)

        # Layer 2: Hidden -> Output
        self.z2 = self.a1 @ self.W2 + self.b2  # (N, output_size)
        self.a2 = sigmoid(self.z2)             # (N, output_size)

        return self.a2

    def backward(self, X, y):
        """Backward pass: compute gradients via backpropagation."""
        m = X.shape[0]  # number of samples

        # Output layer gradient
        # dL/da2 = -(y/a2 - (1-y)/(1-a2)) = (a2 - y) / (a2 * (1-a2))
        # Combined with sigmoid derivative: dL/dz2 = a2 - y
        dz2 = self.a2 - y                              # (N, 1)
        dW2 = (1 / m) * (self.a1.T @ dz2)              # (hidden_size, 1)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

        # Hidden layer gradient (chain rule)
        da1 = dz2 @ self.W2.T                           # (N, hidden_size)
        dz1 = da1 * tanh_derivative(self.a1)            # (N, hidden_size)
        dW1 = (1 / m) * (X.T @ dz1)                    # (input_size, hidden_size)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)  # (1, hidden_size)

        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, learning_rate):
        """Update weights using computed gradients."""
        self.W1 -= learning_rate * self.grads['W1']
        self.b1 -= learning_rate * self.grads['b1']
        self.W2 -= learning_rate * self.grads['W2']
        self.b2 -= learning_rate * self.grads['b2']

    def predict(self, X):
        """Predict binary labels."""
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

    def train(self, X, y, epochs=1000, lr=0.5):
        """Full training loop returning the epoch number where 100% accuracy is reached."""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            self.update(lr)

            accuracy = np.mean(self.predict(X) == y)
            if accuracy == 1.0:
                return epoch + 1

        return None


# ============================================================
# Main: Solve the XOR problem
# ============================================================

def main():
    print("=" * 60)
    print("  Task 1: Experimenting with Hidden Layer Size on XOR")
    print("=" * 60)
    print()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    hidden_sizes = [2, 4, 8, 16]
    max_epochs = 2000
    lr = 0.5

    print(f"Testing hidden sizes {hidden_sizes} with max_epochs={max_epochs}, lr={lr}\n")
    print(f"  {'Hidden Size':<15} | {'Epochs to 100% Acc':<22} | {'Final Accuracy':<15}")
    print(f"  {'-'*15}-+-{'-'*22}-+-{'-'*15}")

    for size in hidden_sizes:
        nn = NeuralNetwork(input_size=2, hidden_size=size, output_size=1)
        epochs_needed = nn.train(X, y, epochs=max_epochs, lr=lr)
        
        final_acc = np.mean(nn.predict(X) == y)
        
        epoch_str = f"{epochs_needed}" if epochs_needed is not None else f"> {max_epochs}"
        print(f"  {size:<15} | {epoch_str:<22} | {final_acc:.2%}")

if __name__ == "__main__":
    main()

