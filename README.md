
# Deep Learning and Information Bottleneck Principle
This repository contains the code of the paper "A Probabilistic Representation for Deep Learning:Delving into The Information Bottleneck Principle" submitted to NeurIPS 2021. 

## Prerequisites
Python 3.7

Tensorflow 1.15

Keras 2.2.4

Numpy

Matplotlib

Please note that the code is senstive to the version of the above packages. To make sure the code can be successfully reimpelmented, we recommond to create an anaconda virtual environment, in which install all the packages with the aforementioned versions.

All the codes are placed in the three folders: (i) Simulations on the bechmark dataset, (ii) Comparison to non-parametric models, and (iii) Simulations on the bechmark dataset.

## Simulations on the bechmark dataset
### test_MLP_Gibbs_IT.py 
It generates the Figure 3, Figure 4, and Figure 5 in the paper.

Please change the value of the parameter 'n_filters' and 'n_neurons' to generate a DNN with different nuerons in the first hidden layer and the second hidden layer.

Please change the value of the parameter 'save_model_index' to visualize the learned weights in different epochs.

## Comparison to non-parametric models

## Simulations on the bechmark dataset, namely MNIST and Fashion-MNIST

### test_MLP_Gibbs_IT_MNIST.py
It generates the results in Appendix H.1

### test_MLP_Gibbs_IT_FMNIST.py
It generates the results in Appendix H.2


