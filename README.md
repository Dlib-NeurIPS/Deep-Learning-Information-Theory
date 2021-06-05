
# Deep Learning and Information Bottleneck Principle
This repository contains the code of the paper "A Probabilistic Representation for Deep Learning:Delving into The Information Bottleneck Principle" submitted to NeurIPS 2021. 

## Gibbs distribution explanation
### test_MLP_Gibbs.py has two functionalities
1. Generating four synthetic images (i.e., image0, image1, image2, image3), which are sampled from a Guassian distribution and sorted by two diagonal directions in the ascedning or descending order. Each image dimension is 32 * 32.

![synthetic_mlp](Simulations/Img_synthetic_MLP.png)

We design a MLP with two fully connected hidden layers. The first hidden layer has 12 neurons, and each neuron has 1024 weights. We reshape the dimension of the weights as 32 * 32 and visualize the weights of all the neurons in the first hidden layer in the above picture.

2. Showing the distribution of the first fully connected hidden layer as Gibbs distribution.

![mlp_f1_gibbs](Simulations/Img_MLP_F1_Gibbs.png)




## Entropy and Mutal information calculation
