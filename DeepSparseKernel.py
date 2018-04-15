import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0.0)

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

    def predict(self, w, x):
        dim, num_data = x.shape
        out           = x;
        bias          = np.ones((1, num_data));
        prev_size     = dim
        start_idx     = 0;
        for i in range(self.num_layers):
            layer_size   = self.layer_sizes[i]
            num_w_layer  = (prev_size+1) * layer_size;
            w_layer      = np.reshape(w[start_idx:start_idx+num_w_layer], (prev_size+1, layer_size))
            out          = self.activation[i](np.dot(w_layer.T, np.concatenate((bias, out))))
            prev_size    = layer_size
            start_idx   += num_w_layer
        return out


def chol_solve(L, y):
    """
    K = L.dot(L.T)
    return inv(K) * y
    """
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)


def chol_inv(L):
    return chol_solve(L, np.eye(L.shape[0]))

class DSK_GP:
    def __init__(self, train_x, train_y, layer_sizes, activations):
        self.train_x   = np.copy(train_x)
        self.train_y   = np.copy(train_y)
        self.dim       = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.nn        = NN(self.dim, layer_sizes, activations)
        self.train_y.reshape(1, train_y.size)
        self.num_param = self.nn.num_param() + 1
        self.nlz       = np.inf

    def log_likelihood(self, theta):
        # TODO: verification of this log_likelihood
        log_sn       = theta[0]
        w            = theta[1:]
        sn2          = np.exp(2 * log_sn)
        Phi          = self.nn.predict(w, self.train_x)
        m, num_train = Phi.shape
        A            = sn2 * np.eye(m) + np.dot(Phi, Phi.T)
        LA           = np.linalg.cholesky(A)

        # data fit
        data_fit_1 = np.dot(self.train_y, self.train_y.T)
        data_fit_2 = np.dot(Phi, self.train_y.T)
        data_fit_2 = chol_solve(LA, data_fit_2)
        data_fit_2 = np.dot(Phi.T, data_fit_2)
        data_fit_2 = np.dot(self.train_y, data_fit_2)
        data_fit   = (data_fit_1 - data_fit_2) / sn2;

        # model complexity
        s, logDetA       = np.linalg.slogdet(A)
        model_complexity = (num_train - m) * (2 * log_sn) + logDetA;

        neg_likelihood   = 0.5 * (data_fit + model_complexity)
        if(neg_likelihood < self.nlz):
            self.nlz   = neg_likelihood
            self.theta = theta
        print(self.nlz.reshape(self.nlz.size)[0])
        return neg_likelihood

    def fit(self, theta):
        theta0     = theta.copy()
        loss       = self.log_likelihood
        gloss      = grad(loss)
        # fmin_cg(loss, theta0, gloss, maxiter = 100)
        fmin_l_bfgs_b(loss, theta0, gloss, maxfun = 500, m=100, iprint=50)

        # pre-computation
        log_sn  = self.theta[0]
        sn2     = np.exp(2 * log_sn)
        w       = self.theta[1:]
        Phi     = self.nn.predict(w, self.train_x)
        m       = Phi.shape[0]
        A       = sn2 * np.eye(m) + np.dot(Phi, Phi.T)
        LA = np.linalg.cholesky(A)

        alpha      = self.train_y.copy()
        alpha      = Phi.dot(alpha.T)
        alpha      = chol_solve(LA, alpha)
        alpha      = Phi.T.dot(alpha)
        alpha      = self.train_y.T - alpha;
        alpha      = alpha / sn2;
        alpha      = Phi.dot(alpha)
        self.alpha = alpha

        Qmm    = Phi.dot(Phi.T)
        self.B = (Qmm - Qmm.dot(chol_solve(LA, Qmm))) / sn2

    def predict(self, test_x):
        log_sn   = self.theta[0]
        w        = self.theta[1:]
        Phi_test = self.nn.predict(w, test_x)
        py       = Phi_test.T.dot(self.alpha)
        ps2      = np.diagonal(np.exp(2 * log_sn) + Phi_test.T.dot(Phi_test) - Phi_test.T.dot(self.B.dot(Phi_test)))
        np.savetxt('sf2', np.diagonal(Phi_test.T.dot(Phi_test)))
        return py, ps2


# def f(x):
#     xsum = x.sum(axis=0);
#     # return np.sign(xsum)
#     return 0.05 * xsum**2 + np.sin(xsum);

# dim       = 1
# num_train = 600
# num_test  = 1000
# sn        = 1e-1
# train_x   = 5 * np.random.randn(dim, num_train)
# train_y   = f(train_x) + sn * np.random.randn(1, num_train)
# test_x    = np.linspace(-30, 60, num_test).reshape(dim, num_test);
# test_y    = f(test_x).reshape(1, num_test);
# gp        = DSK_GP(train_x, train_y, [50, 50], [tanh, tanh])


# theta   = np.random.randn(gp.num_param)
# gloss   = grad(gp.log_likelihood)
# g_theta = gloss(theta)

# gp.fit(theta)
# py, ps2 = gp.predict(test_x)

# plt.plot(test_x[0], py.T[0], 'b')
# plt.plot(test_x[0], py.T[0] + np.sqrt(ps2.T[0]), 'g')
# plt.plot(test_x[0], py.T[0] - np.sqrt(ps2.T[0]), 'g')
# plt.plot(train_x, train_y, 'r+')
# plt.show()

# np.savetxt('train_x', train_x)
# np.savetxt('train_y', train_y)
# np.savetxt('test_x', test_x)
# np.savetxt('test_y', test_y)
# np.savetxt('pred_y', py)
# np.savetxt('pred_s', np.sqrt(ps2))
# np.savetxt('theta', gp.theta)
