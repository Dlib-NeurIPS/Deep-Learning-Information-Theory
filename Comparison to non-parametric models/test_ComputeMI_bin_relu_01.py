
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy.matlib
import numpy as np
import random

from tensorflow import keras
import keras.backend as K

import kde
import simplebinmi

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
#activation_func = 'tanh'
case = 1

# Hyper-parameters
batch_size = 512                # Total number of training images in a single batch
num_epochs = 1001               # Total number of training steps
num_samples = 50                # Total number of training procedures for derive the parameters

binsize = 1.5
def bin_mi(activation,y_label_hot,binsize=1.5):
    n_classes = y_label_hot.shape[1]
    label_mark = np.argmax(y_label_hot,axis=1)
    saved_labelixs = {}
    for i in range(n_classes):
        saved_labelixs[i] = label_mark == i

    MIX_F, MIY_F = simplebinmi.bin_calc_information2(saved_labelixs, activation, binsize)
    return MIX_F,MIY_F

def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

###########################################################################################
# Generating the synethic dataset
x_batch = numpy.zeros((batch_size,img_h,img_w))

def sigmoid(x):
    return 1/(1+np.exp(-x))

image_base = numpy.zeros((img_w,img_h))
for i in range(img_h):
    a1 = np.diag(image_base,i)
    step = float(img_h/(len(a1)))
    b1 = np.zeros(a1.shape)
    for k in range(len(a1)):
        #b1[k] = sigmoid(k * step - img_h/2)*img_h
        b1[k] = k * step
    for k in range(len(a1)):
        image_base[k,k+i] = b1[k]

for i in range(-1,-img_h,-1):
    a1 = np.diag(image_base,i)
    step = float(img_h/(len(a1)))
    b1 = np.zeros(a1.shape)
    for k in range(len(a1)):
        #b1[k] = sigmoid(k * step - img_h/2)*img_h
        b1[k] = k * step
    for k in range(len(a1)):
        image_base[k-i,k] = b1[k]

x_batch[0] = image_base
x_batch[1] = np.flip(x_batch[0],axis=0)
x_batch[1] = np.flip(x_batch[1],axis=1)
x_batch[2] = np.flip(x_batch[0],axis=1)
x_batch[3] = np.flip(x_batch[0],axis=0)

y_true_batch = np.zeros((batch_size, n_classes))

# generate the synthetic dataset
for j in range(batch_size):
    if j % 4 == 0:
        x_batch[j] = x_batch[0]
        y_true_batch[j] = [1,0]

    elif j % 4 == 1:
        x_batch[j] = x_batch[1]
        y_true_batch[j] = [0,1]

    elif j % 4 == 2:
        x_batch[j] = x_batch[2]
        y_true_batch[j] = [1,0]

    else:
        x_batch[j] = x_batch[3]
        y_true_batch[j] = [0,1]

#Adding Guassian noise with 0 mean and std standard deviation to the x_batch
if Anoise:
    x_batch += numpy.random.normal(loc=np.mean(image_base), scale=std, size=x_batch.shape)

x_batch = normalize_meanstd(x_batch, axis=(1,2))
x_max = np.max(x_batch)
x_min = np.min(x_batch)
print('the max and min of the dataset are (%.2f,%.2f)'%(x_max,x_min))


label_mark = np.argmax(y_true_batch,axis=1)
label_0 = np.argwhere(label_mark == 0)
label_1 = np.argwhere(label_mark == 1)

###############################################################################
# Define the neural network for classifying the synthetic dataset
# The neural network includes two fully connected hidden layers
# We use 'tf.layers.conv2d' to construct the first fully connected hidden layer
# We use 'tf.layers.dense' to construct the second fully connected hidden layer
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float64, shape=[None, img_h, img_w], name='X')
    input_layer = tf.reshape(x, shape=[-1, img_size_flat])
    y = tf.placeholder(tf.float64, shape=[None, n_classes], name='Y')
    ###################################################################################
    # The First Hidden Layer
    fcl1 = tf.layers.dense(inputs=input_layer,units=n_filters,
        activation=None,
        use_bias=False,
        #kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
        kernel_initializer='glorot_uniform',
        name='fcl1')
    fc1 = tf.nn.relu(fcl1)
    #fc1 = tf.nn.sigmoid(fcl1)
    #fc1 = tf.tanh(fcl1)
    ###################################################################################
    # The Second Hidden Layer
    fc2 = tf.layers.dense(inputs=fc1,units=n_neurons,
        activation=tf.nn.relu,
        #activation=tf.nn.sigmoid,
        #activation = tf.tanh,
        use_bias=False,
        #kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
        kernel_initializer='glorot_uniform',
        name='fc2')
    ###################################################################################
    # The output layer
    output_logits = tf.layers.dense(inputs=fc2,units=n_classes,
        activation=None,
        #kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
        kernel_initializer='glorot_uniform',
        name='fc3')
    y_pred = tf.nn.softmax(output_logits, name='Prob')
    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # create saver object
    saver = tf.train.Saver()

###########################################################################################
# Constructing the memory for the synthetic training dataset and the activations of each layer
activation1 = np.zeros((num_samples,num_epochs,batch_size,n_filters))
activation2 = np.zeros((num_samples,num_epochs,batch_size,n_neurons))
activation3 = np.zeros((num_samples,num_epochs,batch_size,n_classes))

kld_vector = np.zeros((num_samples,num_epochs))
acc_vector = np.zeros((num_samples,num_epochs))

##########################################################################################
# Training the defined neural network
# run the session
with tf.Session(graph=g) as sess:
    for j in range(num_samples):
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            feed_dict_train = {x: x_batch, y: y_true_batch}
            activation1[j][i], activation2[j][i], activation3[j][i] = sess.run([fc1,fc2,output_logits], feed_dict=feed_dict_train)
            _, kld_vector[j][i],acc_vector[j][i] = sess.run([optimizer,loss,accuracy], feed_dict=feed_dict_train)
            print("Samples %2d, Epoch %02d, KLD %.2f,Accuracy: %.2f%%" % (j, i, kld_vector[j][i],acc_vector[j][i]*100))


MIFX_gvector = np.zeros((num_epochs,3))
MIFY_gvector = np.zeros((num_epochs,3))

MIFX_kvector = np.zeros((num_epochs,3))
MIFY_kvector = np.zeros((num_epochs,3))

MIFX_evector = np.zeros((num_epochs,3))
MIFY_evector = np.zeros((num_epochs,3))

MIFX_4vector = np.zeros((num_epochs,3))
MIFY_4vector = np.zeros((num_epochs,3))


bin_size1 = 0.001 + 0
bin_size2 = 0.01 + 0
bin_size3 = 0.1 + 0
bin_size4 = 1.0 + 0

'''
bin_size1 = 2
bin_size2 = 4
bin_size3 = 6
bin_size4 = 8
'''

for i in range(num_epochs):
    MIX_F1_gs = 0
    MIX_F2_gs = 0
    MIX_FY_gs = 0
    MIY_F1_gs = 0
    MIY_F2_gs = 0
    MIY_FY_gs = 0

    MIX_F1_ks = 0
    MIX_F2_ks = 0
    MIX_FY_ks = 0
    MIY_F1_ks = 0
    MIY_F2_ks = 0
    MIY_FY_ks = 0

    MIX_F1_es = 0
    MIX_F2_es = 0
    MIX_FY_es = 0
    MIY_F1_es = 0
    MIY_F2_es = 0
    MIY_FY_es = 0

    MIX_F1_es4 = 0
    MIX_F2_es4 = 0
    MIX_FY_es4 = 0
    MIY_F1_es4 = 0
    MIY_F2_es4 = 0
    MIY_FY_es4 = 0
    for j in range(num_samples):

        MIX_F1_bin,MIY_F1_bin = bin_mi(activation1[j][i],y_true_batch,bin_size1)
        MIX_F2_bin,MIY_F2_bin = bin_mi(activation2[j][i],y_true_batch,bin_size1)
        MIX_FY_bin,MIY_FY_bin = bin_mi(activation3[j][i],y_true_batch,bin_size1)

        MIX_F1_gs += MIX_F1_bin
        MIX_F2_gs += MIX_F2_bin
        MIX_FY_gs += MIX_FY_bin
        MIY_F1_gs += MIY_F1_bin
        MIY_F2_gs += MIY_F2_bin
        MIY_FY_gs += MIY_FY_bin

        print('bin_size %.2f: MI(X,F1)=%0.3f, MI(Y,F1)=%0.3f' % (bin_size1,MIX_F1_bin,MIY_F1_bin))
        print('bin_size %.2f: MI(X,F2)=%0.3f, MI(Y,F2)=%0.3f' % (bin_size1,MIX_F2_bin,MIY_F2_bin))
        print('bin_size %.2f: MI(X,FY)=%0.3f, MI(Y,FY)=%0.3f' % (bin_size1,MIX_FY_bin,MIY_FY_bin))
        print('')

        MIX_F1_bin,MIY_F1_bin = bin_mi(activation1[j][i],y_true_batch,bin_size2)
        MIX_F2_bin,MIY_F2_bin = bin_mi(activation2[j][i],y_true_batch,bin_size2)
        MIX_FY_bin,MIY_FY_bin = bin_mi(activation3[j][i],y_true_batch,bin_size2)

        MIX_F1_ks += MIX_F1_bin
        MIX_F2_ks += MIX_F2_bin
        MIX_FY_ks += MIX_FY_bin
        MIY_F1_ks += MIY_F1_bin
        MIY_F2_ks += MIY_F2_bin
        MIY_FY_ks += MIY_FY_bin
    
        print('bin_size %.2f: MI(X,F1)=%0.3f, MI(Y,F1)=%0.3f' % (bin_size2,MIX_F1_bin,MIY_F1_bin))
        print('bin_size %.2f: MI(X,F2)=%0.3f, MI(Y,F2)=%0.3f' % (bin_size2,MIX_F2_bin,MIY_F2_bin))
        print('bin_size %.2f: MI(X,FY)=%0.3f, MI(Y,FY)=%0.3f' % (bin_size2,MIX_FY_bin,MIY_FY_bin))
        print('')

        MIX_F1_bin,MIY_F1_bin = bin_mi(activation1[j][i],y_true_batch,bin_size3)
        MIX_F2_bin,MIY_F2_bin = bin_mi(activation2[j][i],y_true_batch,bin_size3)
        MIX_FY_bin,MIY_FY_bin = bin_mi(activation3[j][i],y_true_batch,bin_size3)

        MIX_F1_es += MIX_F1_bin
        MIX_F2_es += MIX_F2_bin
        MIX_FY_es += MIX_FY_bin
        MIY_F1_es += MIY_F1_bin
        MIY_F2_es += MIY_F2_bin
        MIY_FY_es += MIY_FY_bin

        print('bin_size %.2f: MI(X,F1)=%0.3f, MI(Y,F1)=%0.3f' % (bin_size3,MIX_F1_bin,MIY_F1_bin))
        print('bin_size %.2f: MI(X,F2)=%0.3f, MI(Y,F2)=%0.3f' % (bin_size3,MIX_F2_bin,MIY_F2_bin))
        print('bin_size %.2f: MI(X,FY)=%0.3f, MI(Y,FY)=%0.3f' % (bin_size3,MIX_FY_bin,MIY_FY_bin))
        print('')

        MIX_F1_bin,MIY_F1_bin = bin_mi(activation1[j][i],y_true_batch,bin_size4)
        MIX_F2_bin,MIY_F2_bin = bin_mi(activation2[j][i],y_true_batch,bin_size4)
        MIX_FY_bin,MIY_FY_bin = bin_mi(activation3[j][i],y_true_batch,bin_size4)

        MIX_F1_es4 += MIX_F1_bin
        MIX_F2_es4 += MIX_F2_bin
        MIX_FY_es4 += MIX_FY_bin
        MIY_F1_es4 += MIY_F1_bin
        MIY_F2_es4 += MIY_F2_bin
        MIY_FY_es4 += MIY_FY_bin

        print('bin_size %.2f: MI(X,F1)=%0.3f, MI(Y,F1)=%0.3f' % (bin_size4,MIX_F1_bin,MIY_F1_bin))
        print('bin_size %.2f: MI(X,F2)=%0.3f, MI(Y,F2)=%0.3f' % (bin_size4,MIX_F2_bin,MIY_F2_bin))
        print('bin_size %.2f: MI(X,FY)=%0.3f, MI(Y,FY)=%0.3f' % (bin_size4,MIX_FY_bin,MIY_FY_bin))
        print('')


    MIFX_gvector[i] = [MIX_F1_gs/num_samples, MIX_F2_gs/num_samples, MIX_FY_gs/num_samples]
    MIFY_gvector[i] = [MIY_F1_gs/num_samples, MIY_F2_gs/num_samples, MIY_FY_gs/num_samples]

    MIFX_kvector[i] = [MIX_F1_ks/num_samples, MIX_F2_ks/num_samples, MIX_FY_ks/num_samples]
    MIFY_kvector[i] = [MIY_F1_ks/num_samples, MIY_F2_ks/num_samples, MIY_FY_ks/num_samples]

    MIFX_evector[i] = [MIX_F1_es/num_samples, MIX_F2_es/num_samples, MIX_FY_es/num_samples]
    MIFY_evector[i] = [MIY_F1_es/num_samples, MIY_F2_es/num_samples, MIY_FY_es/num_samples]

    MIFX_4vector[i] = [MIX_F1_es4/num_samples, MIX_F2_es4/num_samples, MIX_FY_es4/num_samples]
    MIFY_4vector[i] = [MIY_F1_es4/num_samples, MIY_F2_es4/num_samples, MIY_FY_es4/num_samples]

    np.savez('MLP_IT_bin_%1d_%s.npz'%(case,activation_func),train_acc=acc_vector,kld=kld_vector,MIFX_c1=MIFX_gvector,MIFY_c1=MIFY_gvector,MIFX_c2=MIFX_kvector, MIFY_c2=MIFY_kvector,MIFX_c3=MIFX_evector,MIFY_c3=MIFY_evector,MIFX_c4=MIFX_4vector,MIFY_c4=MIFY_4vector)
    




