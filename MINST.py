import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the parameter grid for grid search
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# Create a KNeighborsClassifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Evaluate the model on the test data
accuracy = grid_search.score(X_test, y_test)

# Print the test accuracy
print("Test accuracy: ", accuracy)


#Best hyperparameters:  {'n_neighbors': 3, 'weights': 'distance'}
#Test accuracy:  0.9728571428571429