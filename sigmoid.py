import numpy as np

# Existing sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Additional activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

# Given data
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Printing results for each activation function
print("Sigmoid Results:", sigmoid(random_values))
print("ReLU Results:", relu(random_values))
print("Leaky ReLU Results:", leaky_relu(random_values))
print("Tanh Results:", tanh(random_values))
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
print("Sigmoid Results:", sigmoid(random_values))
