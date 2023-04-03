import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat), axis=1))


def one_hot_encode(y):
    n_classes = len(np.unique(y))
    return np.eye(n_classes)[y]


def accuracy(y, y_hat):
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def batch_gradient_descent(X_train, y_train, X_val, y_val, n_classes, n_features, n_epochs, learning_rate, early_stop_patience):
    # Initialize weights
    np.random.seed(42)
    weights = np.random.randn(n_features, n_classes)

    # Keep track of loss and accuracy on training and validation sets
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Initialize variables for early stopping
    best_val_loss = np.inf
    epochs_since_last_improvement = 0

    for epoch in range(n_epochs):
        # Forward pass
        y_hat_train = softmax(X_train.dot(weights))

        # Compute loss and accuracy on training set
        train_loss = cross_entropy(y_train, y_hat_train)
        train_acc = accuracy(y_train, y_hat_train)

        # Compute gradients
        grad = X_train.T.dot(y_hat_train - y_train)

        # Update weights
        weights -= learning_rate * grad

        # Forward pass on validation set
        y_hat_val = softmax(X_val.dot(weights))

        # Compute loss and accuracy on validation set
        val_loss = cross_entropy(y_val, y_hat_val)
        val_acc = accuracy(y_val, y_hat_val)

        # Keep track of losses and accuracies
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement == early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return weights, train_losses, val_losses, train_accs, val_accs


# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode labels
y_one_hot = one_hot_encode(y)

# Split dataset into training and validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Run batch gradient descent with early stopping
n_classes = y_one_hot.shape[1]
n_features = X.shape[1]
n_epochs = 1000
learning_rate = 0.0008
early_stop_patience = 20

weights, train_losses, val_losses, train_accs, val_accs = batch_gradient_descent(X_train, y_train, X_val, y_val,
                                                                                  n_classes, n_features, n_epochs,
                                                                                  learning_rate, early_stop_patience)

# Plot training and validation losses
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.show()

# Plot training and validation accuracies
plt.plot(train_accs, label="Training accuracy")
plt.plot(val_accs, label="Validation accuracy")
plt.legend()
plt.show()

# Compute accuracy on test set
y_hat_test = softmax(X_test.dot(weights))
test_acc = accuracy(y_test, y_hat_test)
print(f"Test accuracy: {test_acc:.4f}")

