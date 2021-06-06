
from __future__ import absolute_import, division, print_function

import tensorflow as tf
# Helper libraries

import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Data Dimensions
img_h = img_w = 32              # MNIST images are 32x32
img_size_flat = img_h * img_w   # 28x28=784, the total number of pixels
n_filters = 8					# Number of linear filters
n_neurons = 6					# Number of neurons in the second hidden layer
n_classes = 2                  	# Number of classes


filter_w = 32					# 11 or 32
filter_h = 32					# 11 or 32

learning_rate = 0.03

activation_func = 'relu'
#activation_func = 'tanh'

#If Anoise is True, Gaussian noise will be added, othewise the noise will not be added
Anoise = True

# Hyper-parameters
batch_size = 512                 # Total number of training images in a single batch
num_epochs = 129                 # Total number of training epochs
num_samples = 5 				 # Total number of training iterations, i.e., we will train a network 'num_samples' times, and each time consists of 'num_epochs' training epochs


###############################################################################
# The function 'normalize_meanstd' is to preprocessing the dataset to make sure the dataset with zero mean and one standard derivation
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

###############################################################################
# The function 'image_diagonal' is designed to generate a synthetic image
# We use 'image_diagonal' function to generate four synthetic images, the pixels of a single image are sorted by
# two diagonal directions in the ascedning or descending order.
def image_diagonal(image,lr=False,up=False):
	weight = image
	#print(weight)

	for i in range(img_h):
		a1 = np.diag(weight,i)
		#print(a1)
		b1 = np.sort(a1)
		#print(b1)
		for index in range(img_h-i):
			weight[index,index+i] = b1[index]

	for j in range(-1,-img_h,-1):
		a1 = np.diag(weight,j)
		#print(a1)
		b1 = np.sort(a1)
		#print(b1)
		for index in range(img_h+j):
			weight[index-j,index] = b1[index]

	#print(weight)
	if lr == True:
		weight = np.fliplr(weight)

	if up == True:
		weight = np.flipud(weight)

	return weight

###############################################################################
# Calculate the entropy of given distribution 'pdf'
def cal_entropy(pdf):
    # Guarantee there is no zero proablity
    pdf1 = np.transpose(pdf + np.spacing(1))
    entropy = np.sum(np.multiply(pdf1, np.log2(1/pdf1)),axis=0)
    return entropy

###############################################################################
# Derive the Gibbs distribution based the activations of a hidden layer
def Gibbs_pdf(energy):
    energy = np.float64(energy)
    exp_energy = np.exp(energy+np.spacing(1))
    partition = np.sum(exp_energy,axis=1)
    gibbs = exp_energy/(partition[:,None])
    gibbs = np.float32(gibbs)
    return gibbs

###############################################################################
# Calculate the mutual information between F and Y
def cal_MIFY(pdf_all,y_true_batch):
    label_mark = np.argmax(y_true_batch,axis=1)
    label_0 = np.argwhere(label_mark == 0)
    label_1 = np.argwhere(label_mark == 1)

    pdf_F = np.mean(pdf_all,axis=0)
    entropy_F = cal_entropy(pdf_F)

    pdf_fy0 = np.squeeze(pdf_all[label_0,:])
    pdf_FY0 = np.mean(pdf_fy0,axis=0)
    entropy_FY0 = cal_entropy(pdf_FY0)

    pdf_fy1 = np.squeeze(pdf_all[label_1,:])
    pdf_FY1 = np.mean(pdf_fy1,axis=0)
    entropy_FY1 = cal_entropy(pdf_FY1)

    MIFY = entropy_F - (entropy_FY0 + entropy_FY1)/2

    return MIFY

###############################################################################
# Calculate the mutual information between F and X
def cal_MIFX(pdf_all):
    entropy_all = cal_entropy(pdf_all)
    #print(pdf_all[0])
    conditional_entropy_all = np.mean(entropy_all)
    pdf_F = np.mean(pdf_all,axis=0)
    entropy_F = cal_entropy(pdf_F)
    MIFX = entropy_F - conditional_entropy_all
    return MIFX


###########################################################################################
# Generating the synethic dataset
x_batch = numpy.zeros((batch_size,img_h,img_w))

def sigmoid(x):
    return 1/(1+np.exp(-x))

deterministic_image = numpy.zeros((img_w,img_h))
for i in range(img_h):
    a1 = np.diag(deterministic_image,i)
    step = float(img_h/(len(a1)))
    b1 = np.zeros(a1.shape)
    for k in range(len(a1)):
        #b1[k] = sigmoid(k * step - img_h/2)*img_h
        b1[k] = k * step
    for k in range(len(a1)):
        deterministic_image[k,k+i] = b1[k]

for i in range(-1,-img_h,-1):
    a1 = np.diag(deterministic_image,i)
    step = float(img_h/(len(a1)))
    b1 = np.zeros(a1.shape)
    for k in range(len(a1)):
        #b1[k] = sigmoid(k * step - img_h/2)*img_h
        b1[k] = k * step
    for k in range(len(a1)):
        deterministic_image[k-i,k] = b1[k]

x_batch[0] = deterministic_image
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

###########################################################################################
# Adding Gaussian noise to the synethic training dataset
variance = 1
std = np.sqrt(variance)
if Anoise:
    x_batch = x_batch + numpy.random.normal(loc=np.mean(x_batch), scale=std, size=x_batch.shape)
x_batch = normalize_meanstd(x_batch, axis=(1,2))


###########################################################################################
# Generating the synethic testing dataset
std = np.sqrt(variance)
if Anoise:
    x_test = x_batch + numpy.random.normal(loc=np.mean(x_batch), scale=std, size=x_batch.shape)
else:
    x_test = x_batch
x_test = normalize_meanstd(x_test, axis=(1,2))


###########################################################################################
# Visualizing the synethic dataset

fig, _axs = plt.subplots(nrows=1, ncols=4,figsize=(8,2.3))
axs = _axs.flatten()
for i in range(4):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    title_str = 'Image%d' % (i+1)
    axs[i].set_title(title_str)
    aa = axs[i].imshow(x_batch[i],cmap=plt.get_cmap("gray"))
    #fig.colorbar(aa, ax=axs[i])
fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.9,wspace=0.05, hspace=0.06)
l = 0.92
b = 0.12
w = 0.015
h = 1 - 2*b 
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
cb = plt.colorbar(aa, cax=cbar_ax)


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
    saver = tf.train.Saver(max_to_keep=num_epochs)

###########################################################################################
# Constructing the memory for the synthetic training dataset and the activations of each layer
kld_vector = np.zeros((num_samples,num_epochs))
acc_vector = np.zeros((num_samples,num_epochs))
kld_vector2 = np.zeros((num_samples,num_epochs))
acc_vector2 = np.zeros((num_samples,num_epochs))

#MIXO_vector contains the mutual information between X and each layer
MIFX_vector = np.zeros((num_samples,num_epochs,3))
#MIXO_vector contains the mutual information between Y and each layer
MIFY_vector = np.zeros((num_samples,num_epochs,3))
#MIXO_vector contains the mutual information between X_bar and each layer
MIFXO_vector = np.zeros((num_samples,num_epochs,3))

save_epoch_index = [0,1,2,4,8,128,512,1000]


##########################################################################################
# Training the defined neural network
# run the session
with tf.Session(graph=g) as sess:
    for j in range(num_samples):
        sess.run(tf.global_variables_initializer())
        activation1 = np.zeros((num_epochs,batch_size,n_filters))
        activation2 = np.zeros((num_epochs,batch_size,n_neurons))
        activation3 = np.zeros((num_epochs,batch_size,n_classes))

        for i in range(num_epochs):
            feed_dict_train = {x: x_batch, y: y_true_batch}
            activation1[i], activation2[i], activation3[i] = sess.run([fc1,fc2,output_logits], feed_dict=feed_dict_train)
            if i in save_epoch_index:
                saved_path = saver.save(sess, './saved_new_mlp_%04d'%(i))

            _, kld_vector[j,i],acc_vector[j,i] = sess.run([optimizer,loss,accuracy], feed_dict=feed_dict_train)

            feed_dict_test = {x: x_test, y: y_true_batch}
            kld_vector2[j,i],acc_vector2[j,i] = sess.run([loss,accuracy], feed_dict=feed_dict_test)

            print("Samples %02d, Epoch %02d, KLD %.2f,Accuracy: %.2f%% (Train)" % (j,i, kld_vector[j,i],acc_vector[j,i]*100))
            print("Samples %02d, Epoch %02d, KLD %.2f,Accuracy: %.2f%% (Test)" % (j,i, kld_vector2[j,i],acc_vector2[j,i]*100))

            Gibbs_all_f1 = Gibbs_pdf(activation1[i])
            Gibbs_all_f2 = Gibbs_pdf(activation2[i])
            Gibbs_all_f3 = Gibbs_pdf(activation3[i])

            MIFX_vector[j,i] = [cal_MIFX(Gibbs_all_f1), cal_MIFX(Gibbs_all_f2), cal_MIFX(Gibbs_all_f3)]
            MIFY_vector[j,i] = [cal_MIFY(Gibbs_all_f1,y_true_batch), cal_MIFY(Gibbs_all_f2,y_true_batch), cal_MIFY(Gibbs_all_f3,y_true_batch)]
            MIFXO_vector[j,i] = [MIFX_vector[j,i,0]-MIFY_vector[j,i,0],MIFX_vector[j,i,1]-MIFY_vector[j,i,1],MIFX_vector[j,i,2]-MIFY_vector[j,i,2]]
        
            print('    I(X;T1),     I(X;T2), and     I(X;Y_hat) are (%.2f, %.2f, %.2f)'%(MIFX_vector[j,i,0],MIFX_vector[j,i,1],MIFX_vector[j,i,2]))
            print('    I(Y;T1),     I(Y;T2), and     I(Y;Y_hat) are (%.2f, %.2f, %.2f)'%(MIFY_vector[j,i,0],MIFY_vector[j,i,1],MIFY_vector[j,i,2]))
            print('I(X_bar;T1), I(X_bar;T2), and I(X_bar;Y_hat) are (%.2f, %.2f, %.2f)'%(MIFX_vector[j,i,0]-MIFY_vector[j,i,0],MIFX_vector[j,i,1]-MIFY_vector[j,i,1],MIFX_vector[j,i,2]-MIFY_vector[j,i,2]))


Fontsize = 12

np.savez('MLP_IT_Synthetic_%03d_%03d_%s.npz'%(n_filters,n_neurons,activation_func),train_acc=acc_vector,test_acc=acc_vector2,kld=kld_vector,kld2=kld_vector2,MIFX=MIFX_vector,MIFY=MIFY_vector,MIFXO=MIFXO_vector)
# import the graph from the file
imported_graph = tf.train.import_meta_graph('saved_new_mlp_%04d.meta'%(num_epochs-1))
##########################################################################################
# Visualizing four synthetic images and the learned weights of neurons
# Calculating the Gibbs distribution of each hidden layer
with tf.Session(graph=g) as sess:
    # restore the saved vairable
    imported_graph.restore(sess, './saved_new_mlp_%04d'%(num_epochs-1))

    # Obtain the weights of neurons
    gr = tf.get_default_graph()
    fc1_kernel_val = gr.get_tensor_by_name('fcl1/kernel:0').eval()
    fc2_kernel_val = gr.get_tensor_by_name('fc2/kernel:0').eval()
    fc3_kernel_val = gr.get_tensor_by_name('fc3/kernel:0').eval()

    Hx = np.ones((num_epochs))*2
    Hy = np.ones((num_epochs))

    Location = 'lower right'
    fig, _axs = plt.subplots(nrows=1, ncols=4,figsize=(10,2.5))
    axs = _axs.flatten()


    indexp = 0
    l41, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,0],'r--')
    l42, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,1],'g-.')
    l43, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,2],'b')
    #l44, = axs[indexp].plot(range(1,num_epochs+1),Hx,'-*',color='lime')
    axs[indexp].set_xscale('log')
    if activation_func == 'relu':
        axs[indexp].set_xlabel('(C)',fontsize=Fontsize)
    else:
        axs[indexp].set_xlabel('training epoch \n (H)',fontsize=Fontsize)
    #axs[indexp].set_ylabel(r'$I(\bar{X}_S,F_i)$',fontsize=Fontsize)
    axs[indexp].set_ylim(bottom=-0.05,top=1.2)
    #axs[0].legend(handles = [l11, l12, l13], labels = [r'$MI(F_1,X)$', r'$MI(F_2,X)$',r'$MI(F_3,X)$'],fontsize=Fontsize)
    axs[indexp].grid(True)
    axs[indexp].legend(handles = [l41, l42, l43], labels = [r'$I(\bar{X};T_1)$', r'$I(\bar{X};T_2)$',r'$I(\bar{X};\hat{Y})$'],fontsize=Fontsize,loc='upper right')

    indexp = 1
    l21, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFY_vector,axis=0)[:,0],'r--')
    l22, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFY_vector,axis=0)[:,1],'g-.')
    l23, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFY_vector,axis=0)[:,2],'b')
    l24, = axs[indexp].plot(range(1,num_epochs+1),Hy,'-*',color='orange')
    axs[indexp].set_xscale('log')
    if activation_func == 'relu':
        axs[indexp].set_xlabel('(B)',fontsize=Fontsize)
    else:
        axs[indexp].set_xlabel('training epoch \n (G)',fontsize=Fontsize)
    #axs[indexp].set_ylabel(r'$I(Y,F_i)$',fontsize=Fontsize)
    axs[indexp].set_ylim(bottom=-0.05,top=1.05)
    #axs[1].legend(handles = [l21, l22, l23], labels = [r'$MI(F_1,Y)$', r'$MI(F_2,Y)$',r'$MI(F_3,Y)$'],fontsize=Fontsize)
    axs[indexp].grid(True)
    axs[indexp].legend(handles = [l21, l22, l23, l24], labels = [r'$I(Y,T_1)$', r'$I(Y,T_2)$',r'$I(Y,\hat{Y})$',r'$H(Y)$'],fontsize=Fontsize,loc='lower right')


    indexp = 2
    l41, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,0]+np.mean(MIFY_vector,axis=0)[:,0],'r--')
    l42, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,1]+np.mean(MIFY_vector,axis=0)[:,1],'g-.')
    l43, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,2]+np.mean(MIFY_vector,axis=0)[:,2],'b')
    l44, = axs[indexp].plot(range(1,num_epochs+1),Hx,'-*',color='lime')
    axs[indexp].set_xscale('log')
    if activation_func == 'relu':
        axs[indexp].set_xlabel('(C)',fontsize=Fontsize)
    else:
        axs[indexp].set_xlabel('training epoch \n (H)',fontsize=Fontsize)
    #axs[indexp].set_ylabel(r'$I(\bar{X}_S,F_i)$',fontsize=Fontsize)
    axs[indexp].set_ylim(bottom=-0.05,top=2.2)
    #axs[0].legend(handles = [l11, l12, l13], labels = [r'$MI(F_1,X)$', r'$MI(F_2,X)$',r'$MI(F_3,X)$'],fontsize=Fontsize)
    axs[indexp].grid(True)
    axs[indexp].legend(handles = [l41, l42, l43,l44], labels = [r'$I({X},T_1)$', r'$I({X},T_2)$',r'$I({X},\hat{Y})$',r'$H(X)$'],fontsize=Fontsize,loc='upper left')


    indexp = 3
    l12, = axs[indexp].plot(range(1,num_epochs+1),Hx,'-*',color='lime')
    l14, = axs[indexp].plot(range(1,num_epochs+1),Hy,'-*',color='orange')
    l11, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFXO_vector,axis=0)[:,0]+np.mean(MIFY_vector,axis=0)[:,2],'b')
    l13, = axs[indexp].plot(range(1,num_epochs+1),np.mean(MIFY_vector,axis=0)[:,2],'r-.')
    
    axs[indexp].set_xscale('log')
    if activation_func == 'relu':
        axs[indexp].set_xlabel('(D)',fontsize=Fontsize)
    else:
        axs[indexp].set_xlabel('training epoch \n (F)',fontsize=Fontsize)
    #axs[indexp].set_ylabel(r'$I(S,W)$',fontsize=Fontsize)
    axs[indexp].set_ylim(bottom=-0.05,top=2.2)
    #axtwin = axs[indexp].twinx()
    #axtwin.set_ylabel(r'gen($w$)',fontsize=Fontsize)
    #axtwin.set_ylim(-0.05, 2.1)
    #l12, = axtwin.plot(range(1,num_epochs+1),np.mean(kld_vector2,axis=0),'g')
    #axs[0].legend(handles = [l11, l12, l13], labels = [r'$MI(F_1,X)$', r'$MI(F_2,X)$',r'$MI(F_3,X)$'],fontsize=Fontsize)
    axs[indexp].grid(True)
    axs[indexp].legend(handles = [l11,l13,l12,l14], labels = [r'$I(X,T_{MLP})$',r'$I(Y,T_{MLP})$',r'H(X)',r'H(Y)'],fontsize=Fontsize,loc='upper left')
    fig.tight_layout()

    
    fig, _axs = plt.subplots(nrows=1, ncols=n_filters,figsize=(10,1.3))
    axs = _axs.flatten()
    for i in range(n_filters):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        title_str = r'$\omega_{%d}^{(1)} $' % (i+1)
        axs[i].set_title(title_str)
        aa = axs[i].imshow(fc1_kernel_val[:,i].reshape((filter_h,filter_w)),cmap=plt.get_cmap("gray"))
        #fig.colorbar(aa, ax=axs[i])

    fig.subplots_adjust(bottom=0.05, top=0.8, left=0.02, right=0.9,wspace=0.05, hspace=0.06)

    l = 0.92
    b = 0.05
    w = 0.015
    h = 1 - 4*b 

    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 
    cb = plt.colorbar(aa, cax=cbar_ax)
    
    plt.show()
    




