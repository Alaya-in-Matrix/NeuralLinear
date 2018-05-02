import DeepSparseKernel as dsk
from DeepSparseKernel import np
import matplotlib.pyplot as plt

train_x = np.loadtxt('train_x').T
train_y = np.loadtxt('train_y')
train_y = train_y.reshape(1, train_y.size)

test_x = np.loadtxt('test_x').T
test_y = np.loadtxt('test_y')
test_y = test_y.reshape(1, test_y.size)

layer_sizes = [50, 50]
activations = [dsk.tanh, dsk.tanh]
scale       = 1

dim = train_x.shape[0]

gp       = dsk.DSK_GP(train_x, train_y, layer_sizes, activations, bfgs_iter=500, l1=0, l2=0, debug=True);
theta    = scale * np.random.randn(gp.num_param)
theta[0] = np.log(np.std(train_y) / 2)
theta[1] = np.log(np.std(train_y))
for i in range(dim):
    theta[1+i] = np.log(0.5 * (train_x[i, :].max() - train_x[i, :].min()));
gp.fit(theta, optimize=True)
py, ps2             = gp.predict(test_x)
py_train, ps2_train = gp.predict(train_x)

Phi_train = gp.nn.predict(gp.theta[2:], train_x);
Phi_test  = gp.nn.predict(gp.theta[2:], test_x);

np.savetxt('pred_y', py)
np.savetxt('pred_s2', ps2)
np.savetxt('theta', gp.theta)
np.savetxt('Phi_train', Phi_train)
np.savetxt('Phi_test', Phi_test)

plt.plot(test_y.reshape(test_y.size), py.reshape(py.size), 'r.', train_y.reshape(train_y.size), py_train.reshape(train_y.size), 'b.')
plt.show()

gp.debug = True
print(gp.log_likelihood(gp.theta))
np.savetxt('loss', gp.log_likelihood(gp.theta))
