from   DeepSparseKernel  import np
import matplotlib.pyplot as plt
import sys
import DeepSparseKernel  as dsk
import toml

def trans(data):
    if data.ndim == 1:
        return data.reshape(data.size, 1)
    else:
        return data

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations
num_shared_layer     = conf["num_shared_layer"]
num_non_shared_layer = conf["num_non_shared_layer"]
hidden_shared        = conf["hidden_shared"]
hidden_non_shared    = conf["hidden_non_shared"]
l1                   = conf["l1"]
l2                   = conf["l2"]
scale                = conf["scale"]
max_iter             = conf["max_iter"]
K                    = conf["K"]

train_x              = trans(np.loadtxt('train_x')).T
train_y              = trans(np.loadtxt('train_y'))
test_x               = trans(np.loadtxt('test_x')).T
test_y               = trans(np.loadtxt('test_y'))
dim, num_train       = train_x.shape
num_obj              = train_y.shape[1]
num_test             = train_x.shape[1]

shared_layers_sizes     = [hidden_shared]     * num_shared_layer
shared_activations      = [dsk.tanh]          * num_shared_layer
non_shared_layers_sizes = [hidden_non_shared] * num_non_shared_layer
non_shared_activations  = [dsk.tanh]          * num_non_shared_layer

shared_nn      = dsk.NN(shared_layers_sizes, shared_activations)
non_shared_nns = []

for i in range(num_obj):
    non_shared_nns += [dsk.NN(non_shared_layers_sizes, non_shared_activations)]

modsk = dsk.MODSK(train_x, train_y, shared_nn, non_shared_nns, debug=True, max_iter=max_iter, l1=l1, l2=l2)

py, ps2 = modsk.mix_predict(K, test_x, scale=scale);
np.savetxt('pred_y', py);
np.savetxt('pred_s2', ps2);
print("Finished")
