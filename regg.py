import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(7)
X, y = sklearn.datasets.make_classification(n_samples = 100, n_features=2, 
                            n_redundant=0, n_informative=1,
                            n_clusters_per_class=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(X, Y, learning_rate, iterations):
    
    m = X_train.shape[1]
    n = X_train.shape[0]
    
    W = np.zeros((n,1))
    B = 0
    
    cost_list = []
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        # cost function
        cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
        
        # Gradient Descent
        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)
        
        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list

iterations = 2000
learning_rate = 0.1
W, B, cost_list = model(X_train, Y_train, learning_rate = learning_rate, iterations = iterations)


def accuracy(X, Y, W, B):
    
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")


accuracy(X_test, Y_test, W, B)