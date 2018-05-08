from   DeepSparseKernel  import np
import matplotlib.pyplot as plt
import sys
import DeepSparseKernel  as dsk

def trans(data):
    if data.ndim == 1:
        return data.reshape(data.size, 1)
    else:
        return data

train_x = trans(np.loadtxt('train_x')).T
train_y = trans(np.loadtxt('train_y'))
test_x  = trans(np.loadtxt('test_x')).T
test_y  = trans(np.loadtxt('test_y'))

dim, num_train = train_x.shape
num_obj        = train_y.shape[1]
num_test       = train_x.shape[1]

shared_layers_sizes     = [50, 50];
shared_activations      = [dsk.relu, dsk.tanh]
non_shared_layers_sizes = [50, 50];
non_shared_activations  = [dsk.relu, dsk.tanh]

shared_nn      = dsk.NN(shared_layers_sizes, shared_activations)
non_shared_nns = []

for i in range(num_obj):
    non_shared_nns += [dsk.NN(non_shared_layers_sizes, non_shared_activations)]

modsk = dsk.MODSK(train_x, train_y, shared_nn, non_shared_nns, debug=True, max_iter=200, l1=0, l2=0.1)

# theta   = modsk.rand_theta()
# modsk.fit(theta)
# py, ps2 = modsk.predict(test_x)
# np.savetxt('pred_y', py);
# np.savetxt('pred_s2', ps2);
# log_sns, log_sps, log_lscales, ws = modsk.split_theta(modsk.theta)
# scaled_x                          = dsk.scale_x(modsk.train_x, log_lscales)
# Phis                              = modsk.calc_Phi(ws, scaled_x)
# losses                            = np.array(modsk.loss(modsk.theta))

# scaled_test_x = dsk.scale_x(test_x, log_lscales)
# Phis_test     = modsk.calc_Phi(ws, scaled_test_x)
# if(modsk.debug):
#     for i in range(modsk.num_obj):
#         np.savetxt('Phi_train' + str(i), Phis[i])
#         np.savetxt('Phi_test'  + str(i), Phis_test[i])
#     np.savetxt('train_y_mean', modsk.means)
#     np.savetxt('train_y_std',  modsk.stds)
#     np.savetxt('theta',        modsk.theta)
#     np.savetxt('loss',         losses.reshape(losses.size))


py, ps2 = modsk.mix_predict(8, test_x);
np.savetxt('pred_y', py);
np.savetxt('pred_s2', ps2);
print("Finished")
