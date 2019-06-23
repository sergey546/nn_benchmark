import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer

from utils import *

def get_trainer(trainer_name, net, ds, batchlearning):
    if trainer_name == "bp":
        return BackpropTrainer(net, ds, batchlearning = batchlearning, verbose = True)
    elif trainer_name == "dl":
        return DeepBeliefTrainer(net,ds)

def nn_train(pix, labels):
    print("nn_train")

    hidden = eval(find_argv("hidden", "(40,)"))
    epochs = int(find_argv("epochs", "20"))
    bl = eval(find_argv("batch", "False"))
    trainer_name = find_argv("trainer", "bp")

    print("Hidden layers: {}".format(hidden))
    net = buildNetwork(784, *hidden, 10, bias = True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, recurrent=False)
    net.modulesSorted[1].name = "visible"
    ds = SupervisedDataSet(784, 10)

    for (x,l) in zip(pix, labels): 
        y = [0.0] * 10
        y[int(l[0])] = 1.0
        x = [f/255 for f in x] 
        ds.addSample(x,y)
    trainer = get_trainer(trainer_name, net, ds, bl)#, batchlearning = bl, verbose = True)
    trainer.trainEpochs(epochs)
    return net

def nn_test(net, pix, labels):
    ys = np.matrix(labels).transpose()
    m = ys.shape[1] 
    xs = []
    for (x,l) in zip(pix, labels): 
        x = [f/255 for f in x]
        xs.append(x) 
    res = []
    for line in pix:
        res.append(net.activate(line))

    res = np.matrix(res)
    pred = np.apply_along_axis(lambda x: x.argmax(), axis=1, arr = res)
    print((ys == pred).sum()/m)

def nn(train_pix, train_labels, test_pix, test_labels):
    model = nn_train(train_pix, train_labels)
    print("Train set accuracy:")
    nn_test(model, train_pix, train_labels)
    print("Test set accuracy:")
    nn_test(model, test_pix, test_labels)

