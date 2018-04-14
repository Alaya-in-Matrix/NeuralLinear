import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from scipy.optimize import fmin_cg

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0.0)

class Layer:
    def __init__(self, num_unit, activation_func):
        self.num_unit   = num_unit
        self.activation = activation_func

class NN:
    def __init__(self,dim, layer_sizes, activations):
        self.num_layers  = np.copy(len(layer_sizes))
        self.layer_sizes = np.copy(layer_sizes)
        self.activation  = activations
        self.dim         = dim
    def num_param(self):
        xs = [self.dim];
        np = 0;
        for ls in self.layer_sizes:
            xs.append(ls)
        for i in range(self.num_layers):
            np += (1+xs[i]) * xs[i+1]
        return np
    def weights_to_structure(self, w):
        ws = []
        xs = [self.dim];
        np = 0;
        for ls in self.layer_sizes:
            xs.append(ls)
        start_idx = 0;
        for (ls_this, ls_next) in zip(xs, self.layer_sizes):
            num_w      = (ls_this + 1) * ls_next;
            ws.append(w[start_idx:start_idx+num_w].reshape(ls_this+1, ls_next));
            start_idx += num_w;
        return ws

    def structure_to_weights(self, ws):
        w = np.array([]);
        for s in ws:
            w = np.append(w, s.reshape(s.size))
        return w
    def predict(self, w, x):
        num_data  = x.shape[1]
        out       = x;
        bias      = np.ones((1, num_data));
        prev_size = self.dim;
        start_idx = 0;
        for i in range(self.num_layers):
            layer_size   = self.layer_sizes[i]
            num_w_layer  = (prev_size+1) * layer_size;
            w_layer      = np.reshape(w[start_idx:start_idx+num_w_layer], (prev_size+1, layer_size))
            out          = self.activation[i](np.dot(w_layer.T, np.concatenate((out, bias))))
            prev_size    = layer_size
            start_idx   += num_w_layer
        return out

class DSK_GP:
    def __init__(self, train_x, train_y, layer_sizes, activations):
        self.train_x   = np.copy(train_x)
        self.train_y   = np.copy(train_y)
        self.dim       = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.nn        = NN(self.dim, layer_size, activations)
    def log_likelihood(self):
        pass
    def fit(self):
        pass
    def predict(test_x):
        pass
