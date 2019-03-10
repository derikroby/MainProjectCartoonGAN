#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# In[6]:


#import numpy as np
#from perceptron import Perceptron
from random import randint

training_inputs = []
labels = []
for i in range(5000):
        ip=randint(0,999)
        training_inputs.append(ip)
        if ip<100:
            labels.append(0)
        else:
            labels.append(1)


perceptron = Perceptron(1)
perceptron.train(training_inputs, labels)
print(perceptron.weights)

while(True):
    inp=int(input())
    if ( perceptron.predict(inp)):
        print ("Greater than 100")
    else:
        print ("Less than 100")
    if inp==-1:
        break
        


# In[ ]:




