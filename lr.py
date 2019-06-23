import numpy as np
import scipy.optimize as spopt

from ml_common import *
from data import *


def cost(th, x_array, y_array, m):
    lm = 0.1
    th = np.matrix(th)
    x_array = np.matrix(x_array) 
    sm = sigmoid_matrix(x_array * th.transpose())
    ca = cost_all(sm, y_array)
    J = np.sum(ca) / m

    reg = lm / (m*2)*np.sum(np.multiply(th[:,1:], th[:,1:]))
    J += reg

    a = sm - y_array
    b = np.multiply(a, x_array)
    c = np.sum(b, axis = 0) / m     
    c[:,1:] = c[:,1:] + lm/m * th[:,1:]

    grad = np.array(c)[0,:]

    return J,grad

def lr_train(pix, labels):
    thetas = np.matrix(np.zeros((10, 784+1)))
    ys = []
    for l in labels:
        r = [0.0] * 10
        r[int(l[0])] = 1.0
        ys.append(r)
    ys = np.matrix(ys)
    m = ys.shape[0]
    xs = np.c_[ np.ones(m), normalize_data(np.matrix(pix))]
    for i in range(10):
        print("optimize for {}".format(i))
        th = thetas[[i], :]
        y = ys[:, [i]]
        x = xs
        res = spopt.minimize(cost, th, args = (x, y, m), jac = True, options = {"maxiter" : 50})
        thetas[i, :] = res.x
    return thetas

def lr_test(model, pix, labels):
    ys = np.matrix(labels).transpose()
    m = ys.shape[1]
    x_array = np.c_[ np.ones(m), normalize_data(np.matrix(pix))]
    th = model
    sm = sigmoid_matrix(th * x_array.transpose())
    pred = np.apply_along_axis(lambda x: x.argmax(), axis=0, arr = sm)
    print((ys == pred).sum()/m)


def lr(train_pix, train_labels, test_pix, test_labels):
    model = lr_train(train_pix, train_labels)
    print("Train set accuracy:")
    lr_test(model, train_pix, train_labels)
    print("Test set accuracy:")
    lr_test(model, test_pix, test_labels)
