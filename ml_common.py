from numpy import log,exp
import numpy as np

def sigmoid(z):
    z = np.clip(z, -88.0, 700.0)
    return 1.0/(1.0+exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s*(1-s)

def cost_one(p, y):
    eps = 0.00001
    p = np.clip(p, eps, 1.0 - eps)
    return -y*log(p) - (1-y)*log(1-p)
 
sigmoid_matrix = np.vectorize(sigmoid, otypes = [np.float32])
sigmoid_grad_matrix = np.vectorize(sigmoid_grad, otypes = [np.float32])
cost_all = np.vectorize(cost_one, otypes = [np.float32])
