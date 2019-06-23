convolutional network

depth
h
w
#stride

def conv_pix(p, weights)
    c = 0
    for h,w
        c += sigmoid(p*weights[h,w])
    return c

def conv_1activation(input, output, weights)
    #input net = XxY
    #output net = XxY
    for x,y in X,Y:
        output[x,y] = conv_pix(input[x,y], weights)

class ConvConnection:
    def __init__(self):
        h,w,depth
        weights[h,w]*depth

def forward(self):
    for i in self.depth:
	conv_1activation(layer1,layer2.fmaps[i], weights[i])

def backward(self):
    self.

class ConvLayer:
    def __init__(self):
        self.fmaps = [x,y] * depth

