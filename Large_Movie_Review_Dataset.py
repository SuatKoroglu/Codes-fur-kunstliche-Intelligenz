"""
Task:
In this exercise you will download a dataset, split it, create a tf.data.Dataset
to load it and preprocess it efficiently, then build and train a binary classification
model containing an Embedding layer:
a. Download the Large Movie Review Dataset, which contains 50,000 movie
reviews from the Internet Movie Database (IMDb). The data is organized
in two directories, train and test, each containing a pos subdirectory with
12,500 positive reviews and a neg subdirectory with 12,500 negative reviews.
Each review is stored in a separate text file. There are other files and folders
(including preprocessed bag-of-words versions), but we will ignore them in
this exercise.
b. Split the test set into a validation set (15,000) and a test set (10,000).
c. Use tf.data to create an efficient dataset for each set.
d. Create a binary classification model, using a TextVectorization layer to
preprocess each review.
e. Add an Embedding layer and compute the mean embedding for each review,
multiplied by the square root of the number of words (see Chapter 16). This
rescaled mean embedding can then be passed to the rest of your model.
f. Train the model and see what accuracy you get. Try to optimize your pipelines
to make training as fast as possible.
g. Use TFDS to load the same dataset more easily: tfds.load("imdb_reviews").
"""






from pathlib import Path
import tensorflow as tf
import numpy as np
from pathlib import Path
import chardet
#a
root = "https://ai.stanford.edu/~amaas/data/sentiment/"
filename = "aclImdb_v1.tar.gz"
filepath = tf.keras.utils.get_file(filename, root + filename, extract=True,
                                   cache_dir=".")
path = Path(filepath).with_name("aclImdb")

def tree(path, level=0, indent=4, max_files=3):
    if level == 0:
        print(f"{path}/")
        level += 1
    sub_paths = sorted(path.iterdir())
    sub_dirs = [sub_path for sub_path in sub_paths if sub_path.is_dir()]
    filepaths = [sub_path for sub_path in sub_paths if not sub_path in sub_dirs]
    indent_str = " " * indent * level
    for sub_dir in sub_dirs:
        print(f"{indent_str}{sub_dir.name}/")
        tree(sub_dir,  level + 1, indent)
    for filepath in filepaths[:max_files]:
        print(f"{indent_str}{filepath.name}")
    if len(filepaths) > max_files:
        print(f"{indent_str}...")

def review_paths(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]

train_pos = review_paths(path / "train" / "pos")
train_neg = review_paths(path / "train" / "neg")
test_valid_pos = review_paths(path / "test" / "pos")
test_valid_neg = review_paths(path / "test" / "neg")

len(train_pos), len(train_neg), len(test_valid_pos), len(test_valid_neg)


#b

np.random.shuffle(test_valid_pos)

test_pos = test_valid_pos[:5000]
test_neg = test_valid_neg[:5000]
valid_pos = test_valid_pos[5000:]
valid_neg = test_valid_neg[5000:]

def imdb_dataset(filepaths_positive, filepaths_negative):
    reviews = []
    labels = []
    for filepaths, label in ((filepaths_negative, 0), (filepaths_positive, 1)):
        for filepath in filepaths:
            with open(filepath, 'rb') as review_file:
                raw_data = review_file.read()
                encoding = chardet.detect(raw_data)['encoding']
                reviews.append(raw_data.decode(encoding))
            labels.append(label)
    return tf.data.Dataset.from_tensor_slices(
        (tf.constant(reviews), tf.constant(labels)))

for X, y in imdb_dataset(train_pos, train_neg).take(3):
    print(X)
    print(y)
    print()

#c

def imdb_dataset(filepaths_positive, filepaths_negative, n_read_threads=5):
    dataset_neg = tf.data.TextLineDataset(filepaths_negative,
                                          num_parallel_reads=n_read_threads)
    dataset_neg = dataset_neg.map(lambda review: (review, 0))
    dataset_pos = tf.data.TextLineDataset(filepaths_positive,
                                          num_parallel_reads=n_read_threads)
    dataset_pos = dataset_pos.map(lambda review: (review, 1))
    return tf.data.Dataset.concatenate(dataset_pos, dataset_neg)

batch_size = 32

train_set = imdb_dataset(train_pos, train_neg).shuffle(25000, seed=42)
train_set = train_set.batch(batch_size).prefetch(1)
valid_set = imdb_dataset(valid_pos, valid_neg).batch(batch_size).prefetch(1)
test_set = imdb_dataset(test_pos, test_neg).batch(batch_size).prefetch(1)

#d

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential

# Set the vocabulary size and sequence length
vocab_size = 10000
sequence_length = 500

# Create a TextVectorization layer
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
text_ds = train_set.map(lambda x, y: x)
vectorizer.adapt(text_ds)

# Create the model
model = Sequential([
    vectorizer,
    Embedding(vocab_size, 64),
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

#e

import tensorflow.keras.backend as K

# Create a function to compute the rescaled mean embedding
def rescaled_mean_embedding(x):
    embeddings = model.layers[1](x)
    mask = K.cast(K.not_equal(x, 0), K.floatx())
    embeddings *= K.expand_dims(mask, -1)
    embeddings /= K.sqrt(K.sum(K.square(embeddings), axis=-1, keepdims=True) + K.epsilon())
    mean_embedding = K.sum(embeddings, axis=1) / K.sum(mask, axis=1, keepdims=True)
    return mean_embedding * K.sqrt(K.cast(K.shape(x)[1], K.floatx()))

# Replace the Embedding layer with the rescaled mean embedding layer
model.layers[1] = tf.keras.layers.Lambda(rescaled_mean_embedding)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#f

# Set the number of epochs
epochs = 10

# Set the prefetch buffer size
prefetch_size = tf.data.AUTOTUNE

# Optimize the performance of the datasets
train_ds = train_set.prefetch(prefetch_size)
val_ds = valid_set.prefetch(prefetch_size)
test_ds = test_set.prefetch(prefetch_size)

# Train the model
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# Evaluate the model on the test set
model.evaluate(test_ds)

#g

import tensorflow_datasets as tfds

# Load the dataset
dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

# Split the dataset into train, validation, and test sets
train_ds = dataset["train"]
val_ds = dataset["test"].take(15000)
test_ds = dataset["test"].skip(15000)

# Optimize the performance of the datasets
train_ds = train_ds.batch(batch_size).prefetch(prefetch_size)
val_ds = val_ds.batch(batch_size).prefetch(prefetch_size)
test_ds = test_ds.batch(batch_size).prefetch(prefetch_size)

# Create a TextVectorization layer
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
text_ds = train_ds.map(lambda x, y: x)
vectorizer.adapt(text_ds)

# Create the model
model = Sequential([
    vectorizer,
    Embedding(vocab_size, 64),
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# Evaluate the model on the test set
model.evaluate(test_ds)

"""
Results:

Epoch 1/10
782/782 [==============================] - 15s 16ms/step - loss: 0.6253 - accuracy: 0.7218 - val_loss: 0.5297 - val_accuracy: 0.8050
Epoch 2/10
782/782 [==============================] - 14s 16ms/step - loss: 0.4326 - accuracy: 0.8518 - val_loss: 0.3983 - val_accuracy: 0.8475
Epoch 3/10
782/782 [==============================] - 14s 16ms/step - loss: 0.3337 - accuracy: 0.8813 - val_loss: 0.3391 - val_accuracy: 0.8715
Epoch 4/10
782/782 [==============================] - 14s 16ms/step - loss: 0.2842 - accuracy: 0.8962 - val_loss: 0.3142 - val_accuracy: 0.8767
Epoch 5/10
782/782 [==============================] - 14s 16ms/step - loss: 0.2528 - accuracy: 0.9075 - val_loss: 0.3026 - val_accuracy: 0.8803
Epoch 6/10
782/782 [==============================] - 14s 16ms/step - loss: 0.2282 - accuracy: 0.9172 - val_loss: 0.2921 - val_accuracy: 0.8828
Epoch 7/10
782/782 [==============================] - 15s 17ms/step - loss: 0.2094 - accuracy: 0.9236 - val_loss: 0.2922 - val_accuracy: 0.8827
Epoch 8/10
782/782 [==============================] - 15s 17ms/step - loss: 0.1940 - accuracy: 0.9296 - val_loss: 0.2888 - val_accuracy: 0.8830
Epoch 9/10
782/782 [==============================] - 16s 17ms/step - loss: 0.1800 - accuracy: 0.9352 - val_loss: 0.2911 - val_accuracy: 0.8827
Epoch 10/10
782/782 [==============================] - 15s 17ms/step - loss: 0.1671 - accuracy: 0.9412 - val_loss: 0.2970 - val_accuracy: 0.8821
313/313 [==============================] - 3s 9ms/step - loss: 0.2822 - accuracy: 0.8911
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to C:\Users\Suat\tensorflow_datasets\imdb_reviews\plain_text\1.0.0...
Dl Size...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:14<00:00,  5.40 MiB/s]
Dl Completed...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:14<00:00, 14.81s/ url]
Dataset imdb_reviews downloaded and prepared to C:\Users\Suat\tensorflow_datasets\imdb_reviews\plain_text\1.0.0. Subsequent calls will reuse this data.
Epoch 1/10
778/782 [============================>.] - ETA: 0s - loss: 0.6210 - accuracy: 0.70882023-05-04 05:08:30.264366: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 13s 16ms/step - loss: 0.6206 - accuracy: 0.7093 - val_loss: 0.5221 - val_accuracy: 0.7979
Epoch 2/10
779/782 [============================>.] - ETA: 0s - loss: 0.4260 - accuracy: 0.85082023-05-04 05:08:41.967470: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 12s 15ms/step - loss: 0.4258 - accuracy: 0.8508 - val_loss: 0.3876 - val_accuracy: 0.8583
Epoch 3/10
779/782 [============================>.] - ETA: 0s - loss: 0.3286 - accuracy: 0.88082023-05-04 05:08:53.341722: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 15ms/step - loss: 0.3285 - accuracy: 0.8808 - val_loss: 0.3353 - val_accuracy: 0.8744
Epoch 4/10
778/782 [============================>.] - ETA: 0s - loss: 0.2803 - accuracy: 0.89732023-05-04 05:09:04.682950: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 14ms/step - loss: 0.2802 - accuracy: 0.8973 - val_loss: 0.3101 - val_accuracy: 0.8825
Epoch 5/10
782/782 [==============================] - ETA: 0s - loss: 0.2487 - accuracy: 0.90892023-05-04 05:09:16.076817: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 15ms/step - loss: 0.2487 - accuracy: 0.9089 - val_loss: 0.2965 - val_accuracy: 0.8857
Epoch 6/10
779/782 [============================>.] - ETA: 0s - loss: 0.2250 - accuracy: 0.91882023-05-04 05:09:27.848984: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 12s 15ms/step - loss: 0.2250 - accuracy: 0.9188 - val_loss: 0.2893 - val_accuracy: 0.8878
Epoch 7/10
779/782 [============================>.] - ETA: 0s - loss: 0.2059 - accuracy: 0.92672023-05-04 05:09:39.476923: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 12s 15ms/step - loss: 0.2059 - accuracy: 0.9267 - val_loss: 0.2864 - val_accuracy: 0.8877
Epoch 8/10
779/782 [============================>.] - ETA: 0s - loss: 0.1899 - accuracy: 0.93222023-05-04 05:09:50.899924: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 15ms/step - loss: 0.1899 - accuracy: 0.9322 - val_loss: 0.2866 - val_accuracy: 0.8867
Epoch 9/10
777/782 [============================>.] - ETA: 0s - loss: 0.1761 - accuracy: 0.93802023-05-04 05:10:02.394002: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 15ms/step - loss: 0.1761 - accuracy: 0.9379 - val_loss: 0.2893 - val_accuracy: 0.8861
Epoch 10/10
782/782 [==============================] - ETA: 0s - loss: 0.1640 - accuracy: 0.94282023-05-04 05:10:13.757402: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
782/782 [==============================] - 11s 15ms/step - loss: 0.1640 - accuracy: 0.9428 - val_loss: 0.2938 - val_accuracy: 0.8847
313/313 [==============================] - 2s 6ms/step - loss: 0.2933 - accuracy: 0.8809"""