#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale, normalize
import copy
from sklearn.metrics import log_loss, mean_squared_error
from PIL import Image
from decimal import Decimal
import matplotlib.pyplot as plt
import pickle


# In[2]:


# Utility functions

# Retrieve and pre-process data
def get_data(path):
    X = []
    Y = []
    with h5py.File(path) as dataH5:
        key_list = dataH5.keys()
        for key in key_list:

            matrix = dataH5.get(key)
            for i in range(matrix.shape[0]):
                if key == "X":
                    x = np.asarray(list(matrix[i]))
                    X.append(x)
                else:
                    Y.append(matrix[i])


        Y = [[1,0] if x == 7 else [0,1] for x in Y]
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[1]))
        X = scale(X)
        X = np.array(X, dtype=np.float128)


    return X, Y

# Split into train test val
def train_test_val_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(np.copy(X_train), np.copy(Y_train), test_size=0.2, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


# In[3]:


# Class definitions

# Linear Layers
class Linear:
    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        self.weights = np.random.normal(0, 0.007, (input_dim, output_dim))
        self.bias = np.random.normal(0, 0.007, output_dim)
        
        self.weight_derivative = 0
        self.bias_derivative = 0
        
        self.output = 0
        self.activated_output = 0
        self.activation_function = Sigmoid()
        
        if activation == "sigmoid":
            self.activation_function = Sigmoid()
        elif activation == "softmax":
            self.activation_function = Softmax()
        elif activation == "relu":
            self.activation_function = Relu()
        
    def fc(self, x):
        self.output = np.dot(x, self.weights) + self.bias
        self.activated_output = self.activation_function.activate(self.output)
        return self.activated_output

# Softmax Activation
class Softmax:
    def __init__(self):
        self.output = 0
        self.deriv = 0
        
    def activate(self, x):
        x = np.float128(x)
        for i in range(x.shape[0]):
            if abs(max(x[i])) > abs(min(x[i])):
                x[i][0] = x[i][0] - max(x[i])
                x[i][1] = x[i][1] - max(x[i])

            else:
                x[i][0] = x[i][0] - min(x[i])
                x[i][1] = x[i][1] - min(x[i])
        self.output = np.exp(x) /  np.sum(np.exp(x), axis=1, keepdims=True)
        
        return self.output
    
    def derivative(self):
        pass
    
# ReLu Activation
class Relu:
    def __init__(self, leaky=False):
        self.output = 0
        self.deriv = 0
        self.mode = leaky
    def activate(self, x):
        if self.mode:
            self.output = np.where(x>0, x, 0.05)
        else:
            self.output = np.where(x>0, x, 0)
        return self.output
    
    def derivative(self):
        if self.mode:
            self.deriv = np.where(self.output>0, 1, 0.05)
        else:
            self.deriv = np.where(self.output>0, 1, 0)
        return self.deriv
        

# Sigmoid Activation
class Sigmoid:
    def __init__(self):
        self.output = 0
        self.deriv = 0
        
    def activate(self, x):
        x = np.float128(x)
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def derivative(self):
        self.deriv = self.output * (1 - self.output)
        return self.deriv

# Neg Log Loss / Softmax Loss
class SoftmaxLoss:
    def __init__(self, threshold=0.000000000001):
        self.threshold = threshold
    def loss(self, y_test, y_pred):
        bs = y_test.shape[0]
        
        l = -1/bs*(np.sum(y_test * np.log(y_pred.clip(min=self.threshold))))
        return l

# Neural Network
class PainNN:
    
    def __init__(self, num_hidden_layers, num_neurons, hidden_activation):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.layers = []
        for i in range(num_hidden_layers):
            if hidden_activation == "sigmoid":
                self.layers.append(Linear(self.num_neurons[i], self.num_neurons[i+1], "sigmoid"))
            elif hidden_activation == "relu":
                self.layers.append(Linear(self.num_neurons[i], self.num_neurons[i+1], "relu"))
        self.layers.append(Linear(self.num_neurons[num_hidden_layers], self.num_neurons[self.num_hidden_layers + 1], "softmax"))
        self.num_layers = len(self.layers)

        
    def forward(self, x):
        self.inputs = np.copy(x)
        
        for layer in self.layers:
            x = layer.fc(x)
        
        return x
        
        
    def backward(self, y_test, lr, y_true):
        bs = y_test.shape[0]
        
        self.layers[self.num_layers - 1].deriv = self.layers[self.num_layers - 1].activated_output - y_test
        self.layers[self.num_layers - 1].deriv /= bs
        
        # Output Layer
        error = self.layers[self.num_layers - 1].deriv
        self.layers[self.num_layers-1].weight_derivative = 1/bs*(self.layers[self.num_layers-2].activated_output.T).dot(error)
        self.layers[self.num_layers-1].bias_derivative = 1/bs*(np.sum(error, axis=0))
        
        
        # Hidden Layers - first hidden layer
        for i in range(self.num_layers-2, 0, -1):
            error = np.multiply(error.dot(self.layers[i+1].weights.T), self.layers[i].activation_function.derivative())
            self.layers[i].weight_derivative = 1/bs*(self.layers[i-1].activated_output.T).dot(error)
            self.layers[i].bias_derivative = 1/bs*(np.sum(error, axis=0))
        
        # First Layer
        
        error = np.multiply(error.dot(self.layers[1].weights.T), self.layers[0].activation_function.derivative())
        self.layers[0].weight_derivative = 1/bs*(self.inputs.T).dot(error)
        self.layers[0].bias_derivative = 1/bs*(np.sum(error, axis=0))

        for i in range(self.num_layers):
            self.layers[i].weights -= lr*self.layers[i].weight_derivative
            self.layers[i].bias -= lr*self.layers[i].bias_derivative
        
    


# In[4]:


# Train and Test functions


def train(model, X, Y, X_val, Y_val, num_epochs, lr, bs, loss):

    x_samples = X.shape[0]
    x_val_samples = X_val.shape[0]

    best_model = 0

    min_loss = 1000000000000000000
    max_accuracy = 0
    losses_train = []
    accuracies_train = []
    losses_val = []
    accuracies_val = []

    for epoch in range(num_epochs):

        print("\nEpoch " + str(epoch+1))
        print ("----------")
        
        """
        * Training Phase - both forward and backward phase
        """
        
        train_loss = 0
        train_accuracy = 0
        for i in range(0, x_samples, bs):
            y_ = Y[i:min(i+bs, x_samples)]
            output = model.forward(X[i:min(i+bs, x_samples)])
            y_true = y_.argmax(axis=1)

            model.backward(y_, lr, y_true)
            y_pred = np.argmax(model.layers[model.num_layers - 1].activated_output, axis=1)
            
            train_loss += loss.loss(y_, model.layers[model.num_layers - 1].activated_output)
            train_accuracy += np.sum(y_pred==y_true)

        train_accuracy /= x_samples
        
        

        print ("Training Loss: " + str(train_loss))
        print ("Training Accuracy: " + str(train_accuracy))
        print("\n")
        losses_train.append(train_loss)
        accuracies_train.append(train_accuracy)
        
        """
        * Validation Phase - only forward phase
        """
        
        
        val_loss = 0
        val_accuracy = 0
        for i in range(0, x_val_samples, bs):
            y_ = Y_val[i:min(i+bs, x_val_samples)]
            output = model.forward(X_val[i:min(i+bs, x_val_samples)])
            
            y_pred = np.argmax(model.layers[model.num_layers - 1].activated_output, axis=1)
            
            y_true = y_.argmax(axis=1)
            val_loss += loss.loss(y_, model.layers[model.num_layers - 1].activated_output)
            val_accuracy += np.sum(y_pred==y_true)
            
        val_accuracy /= x_val_samples
        
        if val_loss < min_loss:
            min_loss = val_loss
            
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
        
        print ("Validation Loss: " + str(val_loss))
        print ("Validation Accuracy: " + str(val_accuracy))
        print("\n")
        losses_val.append(val_loss)
        accuracies_val.append(val_accuracy)

    return best_model, losses_train, losses_val, accuracies_val, accuracies_train


def test(model, X, Y, bs):
    x_test_samples = X.shape[0]
    test_accuracy = 0
    test_loss = 0
    for i in range(0, x_test_samples, bs):
        y_ = Y[i:min(i+bs, x_test_samples)]
        output = model.forward(X[i:min(i+bs, x_test_samples)])

        y_pred = np.argmax(model.layers[model.num_layers - 1].activated_output, axis=1)

        y_true = y_.argmax(axis=1)
        test_loss += loss.loss(y_, model.layers[model.num_layers - 1].activated_output)
        test_accuracy += np.sum(y_pred==y_true)

    test_accuracy /= x_test_samples
    return test_accuracy, test_loss


# In[5]:


# Get data and split data
X, Y = get_data("data/MNIST_Subset.h5")

X_train, X_val, X_test, Y_train, Y_val, Y_test = train_test_val_split(X, Y)


# In[6]:


np.random.seed(1)
num_hidden_layers = 1
num_neurons = [28*28, 100, 2]
num_epochs = 1000
model = PainNN(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, hidden_activation="sigmoid")
loss = SoftmaxLoss()
best_model, losses_train, losses_val, accuracies_val, accuracies_train =  train(model, X_train, Y_train, X_val, Y_val, num_epochs=num_epochs, lr=0.05, bs=32, loss=loss)

pickle_out = open("model-1hidden-sigmoid", "wb")
pickle.dump(best_model, pickle_out)
pickle_out.close()


# In[10]:


# Run the best model on the test set

test_accuracy, test_loss = test(best_model, X_test, Y_test, 32)
print (test_accuracy, test_loss)


# In[11]:


# Create training and validation graphs for Sigmoid Activation

x_axis = range(num_epochs)


plt.figure(1)
plt.plot(x_axis, losses_train)
plt.title("Training Loss vs Epochs (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.figure(2)
plt.plot(x_axis, accuracies_train)
plt.title("Training Accuracy vs Epochs (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")

plt.figure(3)
plt.plot(x_axis, losses_val)
plt.title("Validation Loss vs Epochs (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")

plt.figure(4)
plt.plot(x_axis, accuracies_val)
plt.title("Validation Accuracy vs Epochs (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")

plt.show()


# In[12]:


# Train and validate. Change hyperparameters according to validation accuracy/loss - Relu Activation

np.random.seed(1)
num_hidden_layers = 1
num_neurons = [28*28, 100, 2]
num_epochs = 1000
model = PainNN(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, hidden_activation="relu")
loss = SoftmaxLoss()
best_model, losses_train, losses_val, accuracies_val, accuracies_train =  train(model, X_train, Y_train, X_val, Y_val, num_epochs=num_epochs, lr=0.05, bs=32, loss=loss)

pickle_out = open("model-1hidden-relu", "wb")
pickle.dump(best_model, pickle_out)
pickle_out.close()


# In[13]:


# Run the best model on the test set - Relu Activation

test_accuracy, test_loss = test(best_model, X_test, Y_test, 32)
print (test_accuracy, test_loss)


# In[14]:


# Create training and validation graphs for Relu Activation

x_axis = range(num_epochs)


plt.figure(1)
plt.plot(x_axis, losses_train)
plt.title("Training Loss vs Epochs (Relu Activation)")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.figure(2)
plt.plot(x_axis, accuracies_train)
plt.title("Training Accuracy vs Epochs (Relu Activation)")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")

plt.figure(3)
plt.plot(x_axis, losses_val)
plt.title("Validation Loss vs Epochs (Relu Activation)")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")

plt.figure(4)
plt.plot(x_axis, accuracies_val)
plt.title("Validation Accuracy vs Epochs (Relu Activation)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")

plt.show()


# In[ ]:





# In[ ]:




