import numpy as np
from scipy.optimize import minimize
from ml_common import *
from utils import *

class Connection(object):
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        n = layer1.size
        m = layer2.size
        self.weights = np.matrix(np.zeros((n,m)), np.float32)
        self.grad = np.matrix(np.zeros((n,m)), np.float32)

        self.n = n
        self.m = m

    def forward(self):
        self.layer2.zvalues = self.layer1.values * self.weights
        self.layer2.values = sigmoid_matrix(self.layer2.zvalues)

    def init_backward(self):
        np.copyto(self.grad, np.zeros((self.n,self.m)))

    def backward(self):
        self.layer1.deltas = np.multiply(self.layer2.deltas * self.weights.T, sigmoid_grad_matrix(self.layer1.zvalues))
        np.add(self.grad, self.layer1.values.T * self.layer2.deltas, self.grad)

    def finalize_backward(self, m):
        np.copyto(self.grad, self.grad / m)

class Layer(object):
    def __init__(self, size):
        self.size = size
        self.values = np.matrix(np.zeros(size), np.float32)
        self.zvalues = np.matrix(np.zeros(size), np.float32)
        self.deltas = np.matrix(np.zeros(size), np.float32)

class NN(object):
    def __init__(self, structure):
        self.layers = []
        self.connections = []
        self.J = None

        for s in structure:
            layer = Layer(s)
            self.layers.append(layer)
        for layer1,layer2 in pairwise(self.layers):
            connection = Connection(layer1, layer2)
            self.connections.append(connection)

        self.train_input = None
        self.train_output = None
    
    def forward(self):
        for connection in self.connections:
            connection.forward()

    def activate(self, inputs):
        self.layers[0].values = inputs
        self.forward()
        return self.layers[-1].values

    def backward(self):
        for connection in reversed(self.connections):
            connection.backward()

    def wreshape(self, arr):
        ptr = 0
        for connection in self.connections:
            shape = connection.weights.shape
            size = shape[0]*shape[1]
            np.copyto(connection.weights, arr[ptr:ptr+size].reshape(shape))
            ptr += size

    def grad_reshape_1d(self):
        ptr = 0
        for connection in self.connections:
            shape = connection.grad.shape
            size = shape[0] * shape[1]
            np.copyto(self.arr[ptr:ptr+size], connection.grad.reshape(1,size))
            ptr += size
        return self.arr

    def cost(self, weights):
        J = 0.0
        self.wreshape(weights)
        op = self.activate(self.train_inputs)
        deltas = cost_all(op, self.train_outputs)
        J += np.sum(deltas)

        return J/self.train_size

    def get_weights_size(self):
        size = 0
        for connection in self.connections:
            shape = connection.weights.shape
            size += shape[0] * shape[1]
        return size

    def train(self, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.train_size = self.train_inputs.shape[0]
        weights = np.random.rand(self.get_weights_size())
        minimize(self.cost, weights, options = {"maxiter" : 10})

    def costBP(self, weights):
        J = 0.0
        self.wreshape(weights)
        for connection in self.connections:
            connection.init_backward()
        op = self.activate(self.train_inputs)
        deltas = cost_all(op, self.train_outputs)
        J += np.sum(deltas)
        self.layers[-1].deltas = op - self.train_outputs
        self.backward()
        for connection in self.connections:
            connection.finalize_backward(self.train_size)
        grad = self.grad_reshape_1d()
        J = J / self.train_size
        self.J = J
        return J,grad.astype(np.float64)

    def init_train(self):
        pass

    def trainBP(self, train_inputs, train_outputs):
        self.arr = np.ndarray(self.get_weights_size(), np.float32)
        self.train_inputs = train_inputs.astype(np.float32)
        self.train_outputs = train_outputs.astype(np.float32)
        self.train_size = self.train_inputs.shape[0]
        init_weights_scale = 0.01
        weights = np.matrix((np.random.rand(self.get_weights_size()) - 0.5) * init_weights_scale, np.float32)
        minmethod = find_argv("minmethod", "l-bfgs-b")
        nn2method = find_argv("nn2method", "bp")
        nn2iters = int(find_argv("nn2iters", "10"))
        options = {"maxiter" : nn2iters}
        self.init_train()
        if nn2method == "bp":
            minimize(self.costBP, weights, method = minmethod, options = options, jac = True)
        elif nn2method == "grad":
            minimize(self.cost, weights, method = minmethod, options = options, jac = False)

