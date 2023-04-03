from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.interpolation import shift
import numpy as np

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = np.array(mnist['data'], dtype=np.float), mnist['target']

# Convert the data type of X from string to float
X = X.astype(np.float)

# Take a subset of the data to speed up the computations
n_samples = 5000
X_train, X_test, y_train, y_test = X[:n_samples], X[n_samples:n_samples+1000], y[:n_samples], y[n_samples:n_samples+1000]

# Define the function to shift an image
def shift_image(image, direction):
    """Shifts an MNIST image in the given direction (left, right, up, or down) by one pixel."""
    if direction == 'left':
        return shift(image.reshape((28, 28)), [0, -1], cval=0).reshape((-1,))
    elif direction == 'right':
        return shift(image.reshape((28, 28)), [0, 1], cval=0).reshape((-1,))
    elif direction == 'up':
        return shift(image.reshape((28, 28)), [-1, 0], cval=0).reshape((-1,))
    elif direction == 'down':
        return shift(image.reshape((28, 28)), [1, 0], cval=0).reshape((-1,))
    else:
        raise ValueError("Invalid direction. Allowed values: 'left', 'right', 'up', 'down'")

# Augment the training data
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for image, label in zip(X_train, y_train):
    for direction in ['left', 'right', 'up', 'down']:
        shifted_image = shift_image(image, direction)
        X_train_augmented.append(shifted_image)
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Normalize the augmented training data
X_train_augmented = X_train_augmented / 255.0
X_test = X_test / 255.0

# Define the parameter grid for grid search
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# Create a KNeighborsClassifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Fit the model to the augmented training data
grid_search.fit(X_train_augmented, y_train_augmented)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Evaluate the model on the test data
accuracy = grid_search.score(X_test, y_test)

# Print the test accuracy
print("Test accuracy: ", accuracy)
