# README

## About



The last hidden layer of a neural network can be viewed as a finite feature
map, from which a degenerate Gaussian process model can be built; on the other
hands, multiple correlated outputs can be represented by a neural network with
shared hidden layers. In this paper, we build opon these two ideas, and propose
a simple multi-output Gaussian process regression model, the kernels of
multiple outputs are constructed from a multi-task neural network with shared
hidden layers and task-specific layers. We compare our multi-task neural
network enhanced Gaussian process (MTNN-GP) model with several multi-output
Gaussian process models using two public datasets and one examples of
real-world analog integrated circuits, the results show that our model is
competitive compared with these models.

## Future work

- Learning covariance between tasks and handle missing data
- Other architecture: cross-stich
- Advanced NN training: batch-normalization, dropout
- Hyperparameters and architectures of NN: use BO to optimize it
