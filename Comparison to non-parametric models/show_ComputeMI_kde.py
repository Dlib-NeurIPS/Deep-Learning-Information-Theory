
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

activation_func = 'relu'
activation_func = 'tanh'
if activation_func == 'relu':
    print('MLP1(8-6-2 ReLU)')
else:
    print('MLP2(8-6-2 Tanh)')

# Hyper-parameters
batch_size = 512                # Total number of training images in a single batch
num_epochs = 1001               # Total number of training steps
num_samples = 1                # Total number of training procedures for derive the parameters

case = 1
data = np.load('MLP_IT_kde_%1d_%s.npz'%(case,activation_func))

MIFX_gvector1 = data['MIFX_c1']
MIFY_gvector1 = data['MIFY_c1']

MIFX_kvector1 = data['MIFX_c2']
MIFY_kvector1 = data['MIFY_c2']

MIFX_evector1 = data['MIFX_c3']
MIFY_evector1 = data['MIFY_c3']

MIFX_4vector1 = data['MIFX_c4']
MIFY_4vector1 = data['MIFY_c4']

case = 2
data = np.load('MLP_IT_kde_%1d_%s.npz'%(case,activation_func))

MIFX_gvector2 = data['MIFX_c1']
MIFY_gvector2 = data['MIFY_c1']

MIFX_kvector2 = data['MIFX_c2']
MIFY_kvector2 = data['MIFY_c2']

MIFX_evector2 = data['MIFX_c3']
MIFY_evector2 = data['MIFY_c3']

MIFX_4vector2 = data['MIFX_c4']
MIFY_4vector2 = data['MIFY_c4']


 
Fontsize = 13
Location = 'lower right'


noise_variance1 = 0.1 * 0.1
noise_variance2 = 0.1 * 0.5
noise_variance3 = 0.1 * 1
noise_variance4 = 0.1 * 10


fig, _axs = plt.subplots(nrows=2, ncols=4,figsize=(14,6))
axs = _axs.flatten()


indexp = 0
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector1[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector1[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector1[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance1),fontsize=Fontsize)

#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)


indexp = 1
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector1[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector1[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector1[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance2),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 2
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector1[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector1[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector1[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance3),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 3
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector1[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector1[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector1[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance4),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='upper right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)


noise_variance1 = 0.1 * 20
noise_variance2 = 0.1 * 40
noise_variance3 = 0.1 * 80
noise_variance4 = 0.1 * 160


indexp = 4
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector2[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector2[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_gvector2[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance1),fontsize=Fontsize)

#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 5
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector2[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector2[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_kvector2[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance2),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 6
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector2[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector2[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_evector2[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance3),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;T_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)

indexp = 7
l11, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector2[:,0],'r--')
l12, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector2[:,1],'g-.')
l13, = axs[indexp].plot(range(1,num_epochs+1),MIFX_4vector2[:,2],'b')
axs[indexp].set_xscale('log')
axs[indexp].set_xlabel(r'$\sigma_n^2=%.3f$'%(noise_variance4),fontsize=Fontsize)
#axs[indexp].set_ylabel(r'$I(X;F_i)$',fontsize=Fontsize)
axs[indexp].grid(True)
#axs[indexp].set_ylim(0.5, 2.5)
if activation_func == 'relu':
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)
else:
    axs[indexp].legend(handles = [l11, l12, l13], loc='lower right',labels = [r'$I(X;T_1)$', r'$I(X;T_2)$',r'$I(X;\hat{Y})$'],fontsize=Fontsize)


fig.tight_layout()
plt.show()
    




