#!/usr/bin/env python3

from data import *
from utils import *

def main():
    timers.start("dataload")
    train_pix, train_labels, test_pix, test_labels = load_train_data()
    timers.stop("dataload")
    method = find_argv("method", "nn")
    if method == "pybrain":
        import nn_pybrain
        nn_pybrain.nn(train_pix, train_labels, test_pix, test_labels)
    if method == "nn":
        import nn
        nn.nn2(train_pix, train_labels, test_pix, test_labels)
    elif method == "lr":
        import lr
        lr.lr(train_pix, train_labels, test_pix, test_labels)

if __name__ == "__main__":
    main()

