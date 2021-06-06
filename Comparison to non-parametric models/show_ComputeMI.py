
from __future__ import absolute_import, division, print_function

import tensorflow as tf
# Helper libraries

import numpy.matlib
import numpy as np
import random

from tensorflow import keras
import keras.backend as K

import kde
import simplebinmi

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Data Dimensions
img_h = img_w = 32              # MNIST images are 32x32
img_size_flat = img_h * img_w   # 28x28=784, the total number of pixels
n_filters = 8                   # Number of linear filters
n_neurons = 6                   # Number of neurons in the second hidden layer
n_classes = 2                   # Number of classes


filter_w = 32                   # 11 or 32
filter_h = 32                   # 11 or 32

learning_rate = 0.03
Anoise = True
noise_variance = 0.1
std = np.sqrt(noise_variance)


# Hyper-parameters
batch_size = 512                # Total number of training images in a single batch
num_epochs = 1001               # Total number of training steps
num_samples = 3                # Total number of training procedures for derive the parameters

# Functions to return upper and lower bounds on entropy of layer activity
kde_noise_variance = noise_variance*20                 # Added Gaussian noise variance
binsize = 2.0

#activation_func = 'relu'
activation_func = 'tanh'


if activation_func == 'relu':
    data = np.load('MLP_IT_ReLU.npz')
else:
    data = np.load('MLP_IT_Tanh.npz')

acc_vector = data['train_acc']
kld_vector = data['kld']

MIFX_gvector = data['MIFX_gibbs']
MIFY_gvector = data['MIFY_gibbs']

MIFX_kvector = data['MIFX_kde']
MIFY_kvector = data['MIFY_kde']

MIFX_bvector = data['MIFX_bin']
MIFY_bvector = data['MIFY_bin']


Fontsize = 12
Location = 'lower right'

fig, _axs = plt.subplots(nrows=2, ncols=3,figsize=(11,5.5))
axs = _axs.flatten()


'''
indexp = 0
l12, = axs[indexp].plot(range(1,num_epochs+1),1-acc_vector,'m')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel('training epoch\n(A)',fontsize=Fontsize)
axs[indexp].set_ylabel('training error',fontsize=Fontsize)
axs[indexp].grid(True)
#axtwin = axs[indexp].twinx()
#l11, = axs[indexp].plot(range(1,num_epochs+1),1-acc_vector,'c-.')
#axtwin.set_ylabel('training error',fontsize=Fontsize)
#axs[indexp].legend(handles = [l11, l12], labels = ['training error', 'cross entropy'],fontsize=Fontsize)
'''

indexp = 0
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_bvector[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_bvector[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_bvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel('Empirical distribution (bs=2.0)',fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)


indexp = 1
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'KDE ($\sigma_n^2=2.0$)',fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 2
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel('Gibbs distribution',fontsize=Fontsize)
axs[indexp].set_ylim(-0.05, 2.05)
#axs[indexp].set_ylabel(r'$I(X_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
if activation_func =='relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

'''
indexp = 4
l12, = axs[indexp].plot(range(1,num_epochs+1),kld_vector,'c')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel('training epoch \n (E)',fontsize=Fontsize)
axs[indexp].set_ylabel('cross entropy',fontsize=Fontsize)
axs[indexp].grid(True)
'''

indexp = 3
l21, = axs[indexp].plot(range(1,num_epochs+1),MIFY_bvector[:,0],'r--')
l22, = axs[indexp].plot(range(1,num_epochs+1),MIFY_bvector[:,1],'g-.')
l23, = axs[indexp].plot(range(1,num_epochs+1),MIFY_bvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_ylim(-0.05, 1.05)
axs[indexp].set_xlabel('Empirical distribution (bs=2.0)',fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(Y_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
axs[indexp].legend(handles = [l21, l22, l23], loc='lower right',labels = [r'$I(Y;T_1)$', r'$I(Y;T_2)$',r'$I(Y;\hat{Y})$'],fontsize=Fontsize)

indexp = 4
l21, = axs[indexp].plot(range(1,num_epochs+1),MIFY_kvector[:,0],'r--')
l22, = axs[indexp].plot(range(1,num_epochs+1),MIFY_kvector[:,1],'g-.')
l23, = axs[indexp].plot(range(1,num_epochs+1),MIFY_kvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_ylim(-0.05, 1.05)
axs[indexp].set_xlabel(r'KDE ($\sigma_n^2=2.0$)',fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(Y_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
axs[indexp].legend(handles = [l21, l22, l23], loc='lower right',labels = [r'$I(Y;T_1)$', r'$I(Y;T_2)$',r'$I(Y;\hat{Y})$'],fontsize=Fontsize)

indexp = 5
l21, = axs[indexp].plot(range(1,num_epochs+1),MIFY_gvector[:,0],'r--')
l22, = axs[indexp].plot(range(1,num_epochs+1),MIFY_gvector[:,1],'g-.')
l23, = axs[indexp].plot(range(1,num_epochs+1),MIFY_gvector[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_ylim(-0.05, 1.05)
axs[indexp].set_xlabel('Gibbs distribution',fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(Y_S,F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(Y;T_1)$', r'$I(Y;T_2)$',r'$I(Y;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(Y;T_1)$', r'$I(Y;T_2)$',r'$I(Y;\hat{Y})$'],fontsize=Fontsize)


fig.tight_layout()
plt.show()
    




