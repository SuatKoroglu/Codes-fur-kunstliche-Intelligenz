"""
Task: Build your own CNN from scratch and try to achieve the highest possible accuracy on MNIST.
"""

import tensorflow as tf
from keras.callbacks import LearningRateScheduler

# Define the model architecture
model = tf.keras.Sequential(
    [
         # Add a 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten the output of the convolutional layers to a 1D vector
        tf.keras.layers.Flatten(),
        # Add a dense layer with 128 units and ReLU activation
        tf.keras.layers.Dense(units=128, activation="relu"),
        # Add a dense layer with 10 units and softmax activation to output probabilities for each class
        tf.keras.layers.Dense(units=10, activation="softmax"),
    ]
)

# Compile the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load the MNIST dataset and split it into training and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the input data to have a 4D shape of (batch_size, height, width, channels) and normalize pixel values to be between 0 and 1
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert the labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95**x)

# Train the model on the training data for 10 epochs with a batch size of 128 and validation on the test data
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    callbacks=annealer,
    validation_data=(x_test, y_test),
)

# Evaluate the model on the test set and print the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")

"""
Test Loss: 0.031, Test Accuracy: 0.991

How could I improve the code?
-CREATE MORE IMAGES VIA DATA AUGMENTATION
-BUILD MORE CONVOLUTIONAL NEURAL NETWORKS AND ENSEMBLE PREDICTIONS
-ADD BATCH NORMALIZATION AND DROPOUT
"""
