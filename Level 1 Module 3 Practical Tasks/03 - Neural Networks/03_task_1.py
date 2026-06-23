# Task 1: Modify the CNN Architecture
# Extend the CNN practical script by adding a third convolutional layer and increasing the image size to 32x32.
# Generate a dataset with three pattern classes: horizontal lines, vertical lines, and diagonal lines. Train the modified CNN and report the test accuracy.
# Experiment with different numbers of filters (4, 8, 16, 32) in each layer and create a table comparing parameter counts and accuracy.

"""
Practical Script: Simple CNN for Image Classification using PyTorch
Classifies synthetic pattern images (horizontal vs vertical lines)
to demonstrate CNN architecture without needing large datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ============================================================
# Generate synthetic image dataset
# ============================================================

def generate_pattern_dataset(n_samples=500, img_size=32):
    """
    Generate 16x16 grayscale images with two classes:
    Class 0: Horizontal lines
    Class 1: Vertical lines
    """
    images = []
    labels = []

    for _ in range(n_samples):
        img = np.random.uniform(0, 0.1, (1, img_size, img_size))  # Background noise
        label = np.random.randint(0, 3)

        if label == 0:  # Horizontal line
            row = np.random.randint(2, img_size - 2)
            thickness = np.random.randint(1, 3)
            img[0, row:row + thickness, :] = np.random.uniform(0.7, 1.0)
        elif label == 1:  # Vertical line
            col = np.random.randint(2, img_size - 2)
            thickness = np.random.randint(1, 3)
            img[0, :, col:col + thickness] = np.random.uniform(0.7, 1.0)
        else:  # Diagonal line
            shift = np.random.randint(-5, 5)
            diag = np.eye(img_size, k=shift)
            img[0, diag == 1] = np.random.uniform(0.7, 1.0)

        images.append(img)
        labels.append(label)

    return (torch.FloatTensor(images),
            torch.LongTensor(labels))


# ============================================================
# CNN Model
# ============================================================

class SimpleCNN(nn.Module):
    """
    A minimal CNN for binary image classification.

    Architecture:
      Conv(1->8, 3x3) -> ReLU -> MaxPool(2x2)
      Conv(8->16, 3x3) -> ReLU -> MaxPool(2x2)
      Flatten -> Linear(16*3*3, 32) -> ReLU
      Linear(32, 2)
    """
    def __init__(self, num_filters=16):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=3, padding=1),   # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 32x32 -> 16x16

            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 16x16 -> 8x8

            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), # 8x8 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ============================================================
# Training and Evaluation
# ============================================================

def train_cnn(num_filters=16):
    print("=" * 55)
    print("  CNN for Image Classification (PyTorch)")
    print("=" * 55)
    print()

    # Generate data
    X, y = generate_pattern_dataset(n_samples=500, img_size=32)

    # Split train/test
    split = 400
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Image shape:      {X_train[0].shape}")
    print()

    # Create model
    model = SimpleCNN(num_filters=num_filters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    print()

    # Training loop
    print("Training:")
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean()

        if epoch % 3 == 0 or epoch == 14:
            print(f"  Epoch {epoch:2d} | Loss: {loss.item():.4f} | "
                  f"Test Acc: {accuracy:.2%}")

    print()

    # Show some predictions
    print("Sample Predictions:")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test[:10])
        _, predicted = torch.max(test_outputs, 1)

    class_names = ["Horizontal", "Vertical", "Diagonal"]
    for i in range(10):
        true_label = class_names[y_test[i]]
        pred_label = class_names[predicted[i]]
        match = "OK" if y_test[i] == predicted[i] else "WRONG"
        print(f"  Sample {i}: True={true_label:11s}  Pred={pred_label:11s} [{match}]")

    print()
    print("CNN Architecture Summary:")
    print(model)

    return total_params, accuracy.item()

if __name__ == "__main__":
    for filters in [4, 8, 16, 32]:
        train_cnn(num_filters=filters)

