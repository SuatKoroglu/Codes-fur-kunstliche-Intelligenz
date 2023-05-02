"""
Task:
Load the MNIST dataset (introduced in Chapter 3), and split it into a training
set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000
for validation, and 10,000 for testing). Then train various classifiers, such as a
random forest classifier, an extra-trees classifier, and an SVM classifier. Next, try
to combine them into an ensemble that outperforms each individual classifier
on the validation set, using soft or hard voting. Once you have found one, try
it on the test set. How much better does it perform compared to the individual
classifiers?
"""




from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

# Train individual classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma='scale', probability=True, random_state=42)

rf_clf.fit(X_train, y_train)
et_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Evaluate individual classifiers on validation set
for clf in (rf_clf, et_clf, svm_clf):
    y_pred = clf.predict(X_val)
    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))

# Combine classifiers into an ensemble using soft voting
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('et', et_clf), ('svm', svm_clf)],
    voting='soft')

# Train ensemble classifier on training set
voting_clf.fit(X_train, y_train)

# Evaluate ensemble classifier on validation set
y_pred = voting_clf.predict(X_val)
print("Ensemble accuracy:", accuracy_score(y_val, y_pred))

# Evaluate all classifiers on test set
for clf in (rf_clf, et_clf, svm_clf, voting_clf):
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""RandomForestClassifier 0.9692
ExtraTreesClassifier 0.9715
SVC 0.9788
Ensemble accuracy: 0.9791
RandomForestClassifier 0.9645
ExtraTreesClassifier 0.9691
SVC 0.976
VotingClassifier 0.9767"""