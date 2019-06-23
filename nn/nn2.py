import numpy as np
from nn.nn2_np import *
from nn.nn2_ocl import *
from utils import *

def nn2_test(model, pix, labels):
    ys = np.matrix(labels)
    m = ys.shape[0]
    xs = np.matrix(pix, np.float32)
    xs = xs / np.max(xs)
    res = model.activate(xs)
    pr = res.argmax(axis=1)
    acc = np.sum(pr == ys)
    return acc/m

def nn2_train(pix, labels):
    #print("nn2_train")
    hidden = eval(find_argv("hidden", "(40,)"))
    structure = (784,) + hidden + (10,)
    use_ocl = eval(find_argv("ocl", "False"))
    if use_ocl:
        net = CLNN(structure)
    else:
        net = NN(structure)
    xs = np.matrix(pix)
    xs = xs / np.max(xs)
    ys = []
    for l in labels: 
        y = [0.0] * 10
        y[int(l[0])] = 1.0
        ys.append(y)
    ys = np.matrix(ys)
    net.trainBP(xs, ys)
    return net

def nn2(train_pix, train_labels, test_pix, test_labels):
    timers.start("traintime")
    model = nn2_train(train_pix, train_labels)
    timers.stop("traintime")
    timers.start("trainacc")
    train_acc = nn2_test(model, train_pix, train_labels)
    timers.stop("trainacc")
    timers.start("testacc")
    test_acc = nn2_test(model, test_pix, test_labels)
    timers.stop("testacc")
    print(model.J, find_argv("hidden", "(40,)"), find_argv("ocl", "False"), find_argv("minmethod", "l-bfgs-b"), find_argv("nn2method","bp"), find_argv("nn2iters","10"), len(train_labels), len(test_labels), timers.get("dataload"), timers.get("traintime"), timers.get("trainacc"), timers.get("testacc"))

