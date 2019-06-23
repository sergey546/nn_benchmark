import pyopencl as cl
import nn.nn2_np as nn_np
import numpy as np
from utils import pairwise

mf = cl.mem_flags

class CLData(object):
    def __init__(self, ctx):
        self.dev_inv = True
        self.host_inv = True
        self.dev_shape = None
        self.host_shape = None
        self.host_data = None
        self.dev_data = None
        self.ctx = ctx

        self.flags = mf.READ_WRITE

    def host(self):
        return self.get_host()

    def dev(self, *args):
        return self.get_dev(*args)

    def get_host(self):
        if self.host_inv:
            self.dev_to_host()
        return self.host_data

    def get_dev(self, read = True, write = False, shape = None):
        if read and self.dev_inv:
            self.host_to_dev()
        if self.dev_inv or (write and shape != self.dev_shape):
            self.init_dev(shape)
        if write:
            self.host_inv = True
        self.dev_inv = False
        return self.dev_data

    def set_host(self, data):
        self.dev_inv = True
        self.host_inv = False

        self.host_shape = data.shape
        self.host_data = data

    def init_dev(self, shape):
        self.dev_shape = shape
        size = np.prod(self.dev_shape) * 4
        if self.dev_data is not None:
            self.dev_data.release()
        #print("Created buffer ", shape, size)
        self.dev_data = cl.Buffer(self.ctx.ctx, self.flags, size = size)

    def dev_to_host(self):
        #print("d>h")
        if self.dev_inv:
            raise Exception("device data is invalid")
        if (self.host_shape is None) or (self.dev_shape != self.host_shape):
            self.host_shape = self.dev_shape
            self.host_data = np.matrix(np.ndarray(self.host_shape), np.float32)
    
        cl.enqueue_copy(self.ctx.queue, self.host_data, self.dev_data).wait()
        self.host_inv = False

    def host_to_dev(self):
        #print("h>d")
        if self.host_inv:
            raise Exception("host data is invalid")
        if (self.dev_shape is None) or (self.dev_shape != self.host_shape):
            self.init_dev(self.host_shape)
        cl.enqueue_copy(self.ctx.queue, self.dev_data, self.host_data).wait()
        self.dev_inv = False


class CL(object):
    def __init__(self):
        self.ctx = cl.create_some_context(interactive = False)
        src = open("nn/nn_kernels.cl").read()
        self.prg = cl.Program(self.ctx, src).build()
        self.queue = cl.CommandQueue(self.ctx)

    def run_forward(self, connection):
        d_w = connection.weights_dev(True, False)
        d_a1 = connection.layer1.values_dev(True, False)
        
        S = np.int32(connection.layer1.values_proxy.dev_shape[0])
        N = np.int32(connection.weights_proxy.dev_shape[0])
        M = np.int32(connection.weights_proxy.dev_shape[1])
        #print("forward {0}x{1} -> {0}x{2}".format(S,N,M))
        
        d_z2 = connection.layer2.zvalues_dev(False, True, (S,M))
        d_a2 = connection.layer2.values_dev(False, True, (S,M))

        e = self.prg.forward_kernel(self.queue, (S,M), None, d_a1, d_z2, d_a2, d_w, N, M, S).wait()


    def run_backward(self, connection):
        d_d2 = connection.layer2.deltas_dev(True, False)
        d_w = connection.weights_dev(True, False)
        
        
        S = np.int32(connection.layer2.deltas_proxy.dev_shape[0])
        N = np.int32(connection.weights_proxy.dev_shape[0])
        M = np.int32(connection.weights_proxy.dev_shape[1])


        d_a1 = connection.layer1.values_dev(True, False)
        d_z1 = connection.layer1.zvalues_dev(True, False)
        d_d1 = connection.layer1.deltas_dev(False, True, (S,N))

        self.prg.backward_kernel(self.queue, (S,N), None, d_d1, d_d2, d_w, d_z1, N, M, S).wait()

        d_g = connection.grad_dev(False, True, (N, M))
        self.prg.acc_grad(self.queue, (N, M), None, d_d2, d_a1, d_g, N, M, S).wait()

    def run_compute_last_deltas(self, net):
        d_a = net.layers[-1].values_dev(True, False)
        
        S = np.int32(net.layers[-1].values_proxy.dev_shape[0])
        M = np.int32(net.layers[-1].values_proxy.dev_shape[1])

        d_d = net.layers[-1].deltas_dev(False, True, (S,M))
        d_t = net.train_outputs_dev(True, False)

        self.prg.compute_last_deltas(self.queue, (S,M), None, d_a, d_t, d_d, M, S).wait()

    def run_lr_cost(self, net):
        d_a = net.layers[-1].values_dev(True, False)
        S = np.int32(net.layers[-1].values_proxy.dev_shape[0])
        M = np.int32(net.layers[-1].values_proxy.dev_shape[1])
        d_t = net.train_outputs_dev(True, False)
        d_res = net.errors_dev(False, True, (S,M))

        self.prg.lr_cost(self.queue, (S,M), None, d_a, d_t, d_res, M, S).wait()

def proxy(lst):
    def proxy_(cls):
        class deco_class(cls):
            def __init__(self, *args, **kwargs):
                self.proxy_map_host = {}
                self.proxy_map_dev = {}
                self.proxy_map_proxy = {}
                self.proxy_list = []
                super().__init__(*args, **kwargs)
                for n in lst:
                    proxy = CLData(self.ctx)
                    proxy.set_host(cls.__getattribute__(self, n))
                    self.proxy_list.append(proxy)
                    self.proxy_map_host[n] = proxy
                    self.proxy_map_dev[n+"_dev"] = proxy
                    self.proxy_map_proxy[n+"_proxy"] = proxy

            def __getattribute__(self, name):
                if name.startswith("proxy_") or name.startswith("__"):
                    return cls.__getattribute__(self, name)
                if name in self.proxy_map_host:
                    return self.proxy_map_host[name].host()
                elif name in self.proxy_map_dev:
                    return self.proxy_map_dev[name].dev
                elif name in self.proxy_map_proxy:
                    return self.proxy_map_proxy[name]
                return cls.__getattribute__(self, name)

            def __setattr__(self, name, value):
                if name.startswith("proxy_") or name.startswith("__"):
                    self.__dict__[name] = value
                elif name in self.proxy_map_host:
                    self.proxy_map_host[name].set_host(value)
                else:
                    self.__dict__[name] = value

        return deco_class
    return proxy_

@proxy(("values", "zvalues", "deltas"))
class CLLayer(nn_np.Layer):
    def __init__(self, size, ctx = None):
        super().__init__(size)
        self.ctx = ctx

@proxy(("weights", "grad"))
class CLConnection(nn_np.Connection):
    def __init__(self, layer1, layer2, ctx = None):
        super().__init__(layer1, layer2)
        self.ctx = ctx
    
    def forward(self):
        self.ctx.run_forward(self)

    def backward(self):
        self.ctx.run_backward(self)

@proxy(("train_outputs", "train_inputs", "errors"))
class CLNN(nn_np.NN):
    def __init__(self, structure):
        self.layers = []
        self.connections = []
        self.ctx = CL()
        self.train_outputs = np.zeros((0,0))
        self.train_inputs = np.zeros((0,0))
        self.errors = np.zeros((0,0))
        self.J = None

        for s in structure:
            layer = CLLayer(s, ctx = self.ctx)
            self.layers.append(layer)
        for layer1,layer2 in pairwise(self.layers):
            connection = CLConnection(layer1, layer2, ctx = self.ctx)
            self.connections.append(connection)

    def wreshape(self, arr):
        super().wreshape(arr)
        for connection in self.connections:
            connection.weights_proxy.dev_inv = True

    def init_train(self):
        self.layers[0].values = self.train_inputs

    def costBP(self, weights):
        self.wreshape(weights)
        self.forward()
        self.ctx.run_lr_cost(self);
        self.ctx.run_compute_last_deltas(self)
        self.backward()
        grad = self.grad_reshape_1d()
        J = np.sum(self.errors) / self.train_size
        self.J = J
        return J,grad.astype(np.float64)

