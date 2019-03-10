import numpy as np
from math import exp
from random import seed
from random import random,uniform
import gzip
nn_architecture = [
    {"input_dim": 784, "output_dim": 15, "activation": "sigmoid"},
    {"input_dim": 15, "output_dim": 10, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory



def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    
   
    dA_prev = Y - Y_hat   #actual - observed
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx + 1)] += learning_rate * grads_values["dW" + str(layer_idx + 1)]        
        params_values["b" + str(layer_idx + 1)] += learning_rate * grads_values["db" + str(layer_idx + 1)]

    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
    return params_values



import gzip
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
            dataset = np.frombuffer(f.read(), np.uint8, offset=16)

dataset = dataset.reshape(60000,28*28,1)

dataset = dataset/np.float32(256)






n_inputs = 28*28
n_outputs = 10


with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)


output_labels = np.zeros((60000, 10,1))

for i in range(50):
    output_labels[i][labels[i]] = 1

params_values = init_layers(nn_architecture,2)

n_epoch = 50
learning_rate = 0.5
for epoch in range(250):
    sum_error = 0
    
    for i in range(50):
        Y_hat, cashe = full_forward_propagation(dataset[i], params_values, nn_architecture)

        if(epoch > 50):
            print(output_labels[i])
            print(Y_hat)
        error = output_labels[i] - Y_hat
        #print(error)
        for j in range(10):
            sum_error += error[j][0]**2
        grads_values = full_backward_propagation(Y_hat, output_labels[i], cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

    print('====================================\n>epoch=%d, lrate=%.3f, error=%.3f\n===============================' % (epoch, learning_rate, sum_error))

