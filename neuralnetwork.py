# import os
# import numpy as np 
# import glob 
# from PIL import Image, ImageFilter 

# # neural network class based on makeyourownneuralnetwork book
# class neuralNetwork:
#     def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
#         """
#         initializer of neuralNetwork 
#         inputnodes: number of nodes in input layer
#         hiddennodes: number of nodes in hidden layer
#         outputnodes: number of nodes in output layer
#         learningrate: learning rate of neural network
#         """
#         self.nodes_in, self.nodes_hidden, self.nodes_out, self.rate_learning = inputnodes, hiddennodes, outputnodes, learningrate
#         self.weights_hiddenout = np.random.normal(0, (self.nodes_hidden)**(-1/2), (self.nodes_out, self.nodes_hidden))
#         self.weights_inhidden = np.random.normal(0, (self.nodes_in)**(-1/2), (self.nodes_hidden, self.nodes_in))
    
#     def activation_function(self, x):
#         def sigmoid(m):
#             # numpy implementation of sigmoid function
#             return 1.0/(1.0+np.exp(-m))
#         return sigmoid(x)
        
#     def train(self, inputs, targets):
#         """
#         train the neuralNetwork
#         inputs: image value list
#         targets: list of target values
#         """
#         i = np.array(inputs, ndmin=2).T
#         t = np.array(targets, ndmin=2).T
        
#         o_hidden = self.activation_function(np.dot(self.weights_inhidden, i))
#         o_out = self.activation_function(np.dot(self.weights_hiddenout, o_hidden))
#         err_out = t - o_out
#         err_hidden = np.dot(self.weights_hiddenout.T, err_out)
#         temp1, temp2 = o_out * (1.0-o_out) * err_out, o_hidden * (1.0-o_hidden) * err_hidden
#         self.weights_inhidden += np.dot(temp2, np.transpose(i)) * (self.rate_learning)
#         self.weights_hiddenout += np.dot(temp1, np.transpose(o_hidden)) * (self.rate_learning)
        
#     def query(self, inputs):
#         """
#         query the neuralNetwork
#         li: list of input values if uploadedImage is False else image value list
#         uploadedImage: True if querying uploaded image
#         """
#         i = np.array(inputs, ndmin=2).T
#         o_hidden = self.activation_function(np.dot(self.weights_inhidden, i))
#         o_out = self.activation_function(np.dot(self.weights_hiddenout, o_hidden))
#         return o_out
    
#     # save neural network weights
#     def save(self):
#         np.save('digit_wih.npy', self.weights_inhidden)
#         np.save('digit_who.npy', self.weights_hiddenout)

#     # load neural network weights
#     def load(self):
#         self.weights_inhidden = np.load('digit_wih.npy')
#         self.weights_hiddenout = np.load('digit_who.npy')



# # import numpy as np
# # import os
# # import math
# # import _pickle as pickle
# # from scipy import io as spio


# # # %tensorflow_version 1.x
# # import tensorflow as tf

# # # import keras
# # # from keras.datasets import mnist
# # # from keras.models import Sequential
# # # from keras.layers import Dense, Dropout, Flatten
# # # from keras.layers import Conv2D, MaxPooling2D
# # # from keras import backend as K
# # from scipy import io as spio
# # import time 



# # class NeuralNet():
# #     def __init__(self, inodes, hnodes, onodes, layer_num, alpha):
# #         """initialize NN 
# #         inodes: number of nodes in input (number of input features)
# #         hnodes: number of hidden nodes 
# #         onodes: number of nodes in output (possible prediction values: 0-9 in case of digits)
# #         alpha: learning rate
# #         """
# #         self.num_layers = layer_num
# #         self.input_nodes = inodes
# #         self.hidden_nodes = hnodes
# #         self.output_nodes = onodes
# #         self.learning_rate = alpha
# #         self.weights = self.initialize_weights_hidden()

# #     def initialize_weights_hidden(self):
# #         """initialize weights such that it avoids vanishing gradient problem""" 
# #         W = []
# #         wih = np.random.normal(0, np.sqrt(2/self.input_nodes), (self.hidden_nodes, self.input_nodes))
# #         woh = np.random.normal(0, np.sqrt(2/self.hidden_nodes), (self.output_nodes, self.hidden_nodes))
# #         W.append(wih)
# #         for i in range(1,self.num_layers):
# #             W.append(np.random.normal(0, np.sqrt(2/self.hidden_nodes), (self.hidden_nodes, self.hidden_nodes)))

# #         W.append(woh)

# #         return W



# #     def activation_function(self,x):
# #         return 1.0/(1.0+np.exp(-x))

# #     def find_derivative(self,x):
# #         return x * (1-x)

# #     def compute_Loss(self, pred_array, initial_output):
# #         """
# #         Computes loss from forward propagation 
# #         parameters
# #         pred_array: list of possible predictions transposed
# #         initial_output: output from forward propagation
# #         woh: weights out of hidden layer

# #         """
# #         err = pred_array - initial_output

# #         return err

# #     def forward_propagation(self, input_array,index):
# #         """
# #         Predicts output through forward propagation through the neural network
# #         paramaters
# #         input: list of image value
# #         pred_vals: list of possible predictions
# #         """
# #         output_from_hidden = self.activation_function(np.dot(self.weights[index], input_array))


# #         return output_from_hidden

# #     def backprop(self, output, err, index):
# #         derivative = self.find_derivative(output[index])
# #         temp = derivative * err
# #         self.weights[index] += np.dot(temp, np.transpose(output[index-1])) * self.learning_rate

# #         return self.weights[index]



# #     def train(self, input, pred_vals):
# #         input_array = np.array(input, ndmin=2).T
# #         pred_array = np.array(pred_vals, ndmin=2).T
# #         output_from_layer = []
# #         index = 0
# #         #forward propagate throught the hidden layers
# #         for i in range(len(self.weights)): 
# #             output_from_layer.append(self.forward_propagation(input_array,index))
# #             input_array = output_from_layer[index]
# #             index += 1

    

# #         #compute initial loss
# #         err = self.compute_Loss(pred_array,output_from_layer[-1])
# #         # final_err = np.dot(a, err)

# #         W = self.weights

# #         #update weights through back prop
# #         for i in range(len(self.weights)-1,0,-1):
# #             new_err = np.dot(W[i].T , err)
# #             self.weights[i] = self.backprop(output_from_layer,err,i)
# #             err = new_err

# #         derivative = self.find_derivative(output_from_layer[0])
# #         temp = derivative * err
# #         self.weights[0] += np.dot(temp, np.transpose(np.array(input, ndmin=2).T)) * (self.learning_rate)

# #     def save(self, st):
# #         np.save("./weights/0weights" + str(st) + ".npy", self.weights)

# #     def load(self, st):
# #         self.weights = np.load("./weights/0weights" + str(st) + ".npy", allow_pickle= True)


# #     def query(self, inputs, model):
# #         input = np.array(inputs, ndmin=2).T
# #         for i in range(len(model.weights)):
# #             out = model.activation_function(np.dot(model.weights[i], input))
# #             input = out

# #         return input
  



import numpy as np
import os
import math
#import _pickle as pickle
import imageio
# import matplotlib.pyplot
import glob
import cv2
from scipy import ndimage
from PIL import Image, ImageFilter

import keras
from keras.datasets import mnist


class NeuralNet():
  def __init__(self, inodes, hnodes, onodes, alpha, activation_function_type):
    """initialize NN 
    inodes: number of nodes in input (number of input features)
    hnodes: number of hidden nodes 
    onodes: number of nodes in output (possible prediction values: 0-9 in case of digits)
    alpha: learning rate
    """

    self.input_nodes = inodes
    self.hidden_nodes = hnodes
    self.output_nodes = onodes
    self.learning_rate = alpha
    self.activation_function_type = activation_function_type
    self.wih, self.woh = self.initialize_weights_hidden()

  def initialize_weights_hidden(self):
    """initialize weights such that it avoids vanishing gradient problem""" 
    wih = np.random.normal(0, np.sqrt(2/self.input_nodes), (self.hidden_nodes, self.input_nodes))
    woh = np.random.normal(0, np.sqrt(2/self.hidden_nodes), (self.output_nodes, self.hidden_nodes))
  

    return wih, woh


  def activation_function(self, x):
    if self.activation_function_type == "Sigmoid":
      return 1.0/(1.0+np.exp(-x))
    elif self.activation_function_type == "ReLu":
      return np.maximum(0,x)
    elif self.activation_function_type == "Tanh":
      return np.tanh(x)

  def find_derivative(self, x):
    if self.activation_function_type == "Sigmoid":
      return x * (1-x)

    elif self.activation_function_type == "ReLu": 
      if i <= 0:
        return 0
      else:
        return 1
    elif self.activation_function_type == "Tanh":
      return 1 - (self.activation_function(x) ** 2)


  def compute_Loss(self, pred_array, initial_output):
    """
      Computes loss from forward propagation 
      parameters
      pred_array: list of possible predictions transposed
      initial_output: output from forward propagation
      woh: weights out of hidden layer

    """
    err = pred_array - initial_output
    return err, np.dot(self.woh.T, err)

  def forward_propagation(self, input_array, pred_array):
    """
    Predicts output through forward propagation through the neural network
    paramaters
    wih: weights into hidden layer
    woh: weights out of hidden layer
    input: list of image value
    pred_vals: list of possible predictions
    """
    output_from_hidden = self.activation_function(np.dot(self.wih, input_array))
    initial_output = self.activation_function(np.dot(self.woh, output_from_hidden))

    return output_from_hidden, initial_output


  def backprop(self, input_array, pred_array, initial_output, output_from_hidden):
    err, err_from_hidden = self.compute_Loss(pred_array,initial_output)
    derivative_out = self.find_derivative(initial_output)
    derivative_hidden = self.find_derivative(output_from_hidden)
    temp1 = derivative_out * err
    temp2 = derivative_hidden * err_from_hidden
   
    self.wih += np.dot(temp2, np.transpose(input_array)) * (self.learning_rate)
    self.woh += np.dot(temp1, np.transpose(output_from_hidden)) * (self.learning_rate)

    return self.wih, self.woh



  def train(self, input, pred_vals):
    input_array = np.array(input, ndmin=2).T
    pred_array = np.array(pred_vals, ndmin=2).T
    output_from_hidden, initial_output = self.forward_propagation(input_array, pred_array)
    
    self.wih, self.woh = self.backprop(input_array, pred_array, initial_output, output_from_hidden)
    
  def query(self, inputs, model):
    i = np.array(inputs, ndmin=2).T
    o_hidden = model.activation_function(np.dot(model.wih, i))
    o_out = model.activation_function(np.dot(model.woh, o_hidden))
    return o_out
  
  def save(self):
    np.save("./weights/who" + ".npy", self.woh)
    np.save("./weights/wih" + ".npy", self.wih)


  def load(self):
    self.woh = np.load("./weights/digit_who.npy", allow_pickle=True)
    self.wih = np.load("./weights/digit_wih.npy", allow_pickle=True)
