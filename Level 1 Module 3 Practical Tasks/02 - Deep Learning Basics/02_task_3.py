# Task 3: Compare Activation Functions
# Using the XOR network as a base, run experiments comparing different activation functions for the hidden layer: sigmoid, tanh, and ReLU. 
# For each activation function, train for 2000 epochs and record the training loss curve (loss at epoch 0, 500, 1000, 1500, 2000). 
# Create a comparison table and explain which activation function works best for this problem and why.

import numpy as np

# ============================================================
# Activation functions and their derivatives
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh_fn(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)


# ============================================================
# Loss function
# ============================================================

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================
# Neural Network class
# ============================================================

class NeuralNetwork:

    def __init__(self, input_size=2, hidden_size=4, output_size=1, act_type='tanh'):
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros((1, output_size))

        self.act_type = act_type
        if act_type == 'sigmoid':
            self.act, self.act_deriv = sigmoid, sigmoid_derivative
        elif act_type == 'relu':
            self.act, self.act_deriv = relu, relu_derivative
        else:
            self.act, self.act_deriv = tanh_fn, tanh_derivative

        self.grads = {} 

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1       
        self.a1 = self.act(self.z1)             

        self.z2 = self.a1 @ self.W2 + self.b2  
        self.a2 = sigmoid(self.z2)             

        return self.a2

    def backward(self, X, y):
        m = X.shape[0]  

        dz2 = self.a2 - y                                              
        dW2 = (1 / m) * (self.a1.T @ dz2)                              
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)  

        da1 = dz2 @ self.W2.T                                           
        dz1 = da1 * self.act_deriv(self.a1)            
        dW1 = (1 / m) * (X.T @ dz1)                    
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)  

        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.grads['W1']
        self.b1 -= learning_rate * self.grads['b1']
        self.W2 -= learning_rate * self.grads['W2']
        self.b2 -= learning_rate * self.grads['b2']

    def train(self, X, y, epochs=2001, lr=0.5):
        loss_curve = {}
        for epoch in range(epochs):
            output = self.forward(X)
            loss = binary_cross_entropy(y, output)
            self.backward(X, y)
            self.update(lr)
            
            if epoch in [0, 500, 1000, 1500, 2000]:
                loss_curve[epoch] = loss
        return loss_curve


# ============================================================
# Main: Comparative Experiments
# ============================================================

def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    activations = ['sigmoid', 'tanh', 'relu']
    results = {}

    for act in activations:
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, act_type=act)
        results[act] = nn.train(X, y, epochs=2001, lr=0.5)

    print(f"  {'Activation':<12} | {'Epoch 0':<10} | {'Epoch 500':<10} | {'Epoch 1000':<10} | {'Epoch 1500':<10} | {'Epoch 2000':<10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for act in activations:
        curve = results[act]
        print(f"  {act:<12} | {curve[0]:.6f}   | {curve[500]:.6f}   | {curve[1000]:.6f}   | {curve[1500]:.6f}   | {curve[2000]:.6f}")

if __name__ == "__main__":
    main()