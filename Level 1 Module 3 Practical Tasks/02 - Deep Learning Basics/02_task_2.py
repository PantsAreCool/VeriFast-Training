# Task 2: Implement a Regression Network
## Modify the neural network to solve a regression problem instead of classification. 
## Generate a dataset of 200 points from the function y = sin(x) + noise. 
## Replace the sigmoid output activation with a linear (identity) activation, and replace binary cross-entropy with mean squared error loss. 
## Train the network and print predictions for 10 test points alongside the true values.

import numpy as np

# ============================================================
# Activation functions and their derivatives
# ============================================================

def identity(z):
    return z

def identity_derivative(a):
    return np.ones_like(a)

def tanh_fn(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a ** 2


# ============================================================
# Loss function
# ============================================================

def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


# ============================================================
# Neural Network class
# ============================================================

class RegressionNetwork:

    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros((1, output_size))

        self.grads = {} 

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1       
        self.a1 = tanh_fn(self.z1)             

        self.z2 = self.a1 @ self.W2 + self.b2  
        self.a2 = identity(self.z2)             

        return self.a2

    def backward(self, X, y):
        m = X.shape[0]  

        # dL/dy_pred for MSE is 2/m * (y_pred - y). Combined with identity derivative (1):
        dz2 = (2 / m) * (self.a2 - y)                                              
        dW2 = self.a1.T @ dz2                              
        db2 = np.sum(dz2, axis=0, keepdims=True)  

        da1 = dz2 @ self.W2.T                                           
        dz1 = da1 * tanh_derivative(self.a1)            
        dW1 = X.T @ dz1                    
        db1 = np.sum(dz1, axis=0, keepdims=True)  

        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.grads['W1']
        self.b1 -= learning_rate * self.grads['b1']
        self.W2 -= learning_rate * self.grads['W2']
        self.b2 -= learning_rate * self.grads['b2']

    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mean_squared_error(y, output)
            self.backward(X, y)
            self.update(lr)
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | MSE Loss: {loss:.6f}")


# ============================================================
# Main: Train and Evaluate
# ============================================================

def main():
    np.random.seed(42)
    X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    noise = np.random.normal(0, 0.1, X.shape)
    y = np.sin(X) + noise

    print("Training Regression Network...")
    net = RegressionNetwork(input_size=1, hidden_size=10, output_size=1)
    net.train(X, y, epochs=1000, lr=0.1)


    X_test = np.linspace(-np.pi, np.pi, 10).reshape(-1, 1)
    y_true = np.sin(X_test)
    y_pred = net.forward(X_test)

    print("Predictions vs True Values:")
    print(f"  {'X Value':<10} | {'True Y':<10} | {'Predicted Y':<12}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for i in range(10):
        print(f"  {X_test[i,0]:10.4f} | {y_true[i,0]:10.4f} | {y_pred[i,0]:12.4f}")

if __name__ == "__main__":
    main()