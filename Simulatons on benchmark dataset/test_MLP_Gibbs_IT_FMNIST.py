
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std
##############################################################################
### Load MNIST dataset and prepareing the training samples and labels
##############################################################################
dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images = normalize_meanstd(train_images, axis=(1,2))
test_images = normalize_meanstd(test_images, axis=(1,2))

num_train_samples = 60000
num_test_samples = len(test_labels)

train_samples = train_images[:num_train_samples]
train_labels = train_labels[:num_train_samples]
train_labels_onehot = np.zeros((num_train_samples,10))
for i in range(num_train_samples):
	train_labels_onehot[i,train_labels[i]] = 1

test_samples = test_images
test_labels_onehot = np.zeros((num_test_samples,10))
for i in range(num_test_samples):
	test_labels_onehot[i,test_labels[i]] = 1

print(train_labels[:10])
print(train_labels_onehot[0])
print('The train samples dimension are (%d, %d, % d)'%(train_samples.shape[0],train_samples.shape[1],train_samples.shape[2]))
print('The train labels  dimension are (%d, %d)'%(train_labels_onehot.shape[0],train_labels_onehot.shape[1]))
print('The test  samples dimension are (%d, %d, % d)'%(test_samples.shape[0],test_samples.shape[1],test_samples.shape[2]))
print('The test  labels  dimension are (%d, %d)'%(test_labels_onehot.shape[0],test_labels_onehot.shape[1]))


##########################
### SETTINGS
##########################

activation_func = 'relu'
#activation_func = 'tanh'
#activation_func = 'sigmoid'

# Hyperparameters
learning_rate = 0.0005
training_epochs = 500
n_samples = 20

batch_size = 600


# Architecture
img_h = img_w = 28              # FMNIST images are 28x28
n_input = 784
n_classes = 10

n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 96

print(n_hidden_1, n_hidden_2)


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
# Calculate the mutual information between F and X
def cal_MIFX(pdf_all):

    entropy_all = cal_entropy(pdf_all)
    conditional_entropy_all = np.mean(entropy_all)

    pdf_F = np.mean(pdf_all,axis=0)
    pdf_F = (pdf_F/np.sum(pdf_F)).reshape(1,-1)
    #print(pdf_F.shape)
    entropy_F = cal_entropy(pdf_F)
    #print(pdf_F)

    MIFX = entropy_F - conditional_entropy_all
    #print(entropy_F,conditional_entropy_all)
    #print(MIFX)
    #print(3/0)
    #print(entropy.shape)
    return MIFX

###############################################################################
# Calculate the mutual information between F and Y
def cal_MIFY(pdf_all,y_true_batch):

    label_mark = np.argmax(y_true_batch,axis=1)

    pdf_F = np.mean(pdf_all,axis=0)
    pdf_F = (pdf_F/np.sum(pdf_F)).reshape(1,-1)
    entropy_F = cal_entropy(pdf_F)

    H_FY_sum = 0
    for i in range(10):
        labeli = np.argwhere(label_mark == i)
        pdf_fyi = np.squeeze(pdf_all[labeli,:])
        pdf_FYi = np.mean(pdf_fyi,axis=0)
        pdf_FYi = (pdf_FYi/np.sum(pdf_FYi)).reshape(1,-1)
        entropy_FYi = cal_entropy(pdf_FYi)
        #print(np.sum(pdf_FYi))
        H_FY_sum += entropy_FYi

    n_features = pdf_all.shape[1]
    #entropy_F = np.log2(n_features)
    #print('H(F_i) and H(F_i|Y) are %.2f,%.2f' %(entropy_F,H_FY_sum/10))
    MIFY = entropy_F - H_FY_sum/10

    #print(entropy.shape)
    return MIFY


######################################################
### TRAINING the MLP with Adam
######################################################
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(tf.float32, [None, img_h,img_w], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')
    input_layer = tf.reshape(tf_x, shape=[-1, n_input])
    ###################################################################################
    # The First Hidden Layer
    fcl1 = tf.layers.dense(inputs=input_layer,units=n_hidden_1,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.001),
        #kernel_initializer='glorot_uniform',
        name='fcl1')
    fc1 = tf.nn.relu(fcl1)
    #fc1 = tf.nn.sigmoid(fcl1)
    #fc1 = tf.tanh(fcl1)
    ###################################################################################
    # The Second Hidden Layer
    fc2 = tf.layers.dense(inputs=fc1,units=n_hidden_2,
        activation=tf.nn.relu,
        #activation=tf.nn.sigmoid,
        #activation = tf.tanh,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1),
        #kernel_initializer='glorot_uniform',
        name='fc2')
    fc3 = tf.layers.dense(inputs=fc2,units=n_hidden_3,
        activation=tf.nn.relu,
        #activation=tf.nn.sigmoid,
        #activation = tf.tanh,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1),
        #kernel_initializer='glorot_uniform',
        name='fc3')
    ###################################################################################
    # The output layer
    output_logits = tf.layers.dense(inputs=fc3,units=n_classes,
        activation=None,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1),
        #kernel_initializer='glorot_uniform',
        name='fco')
    y_pred = tf.nn.softmax(output_logits, name='Prob')
    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=output_logits), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss, name='train')
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(tf_y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # create saver object
    saver = tf.train.Saver()



##########################
### TRAINING & EVALUATION
##########################
#MIX_vector contains the mutual information between X and each layer
MIFX_vector = np.zeros((n_samples,training_epochs,4))
#MIY_vector contains the mutual information between Y and each layer
MIFY_vector = np.zeros((n_samples,training_epochs,4))
#MIXO_vector contains the mutual information between X_bar and each layer
MIFXO_vector = np.zeros((n_samples,training_epochs,4))

train_acc_vector = np.zeros((n_samples,training_epochs))
test_acc_vector = np.zeros((n_samples,training_epochs))
kld_vector = np.zeros((n_samples,training_epochs))
kld_vector2 = np.zeros((n_samples,training_epochs))


with tf.Session(graph=g) as sess:
    for k in range(n_samples):
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = num_train_samples // batch_size

            activation1, activation2, activation3,activationo = sess.run([fc1, fc2, fc3,output_logits], feed_dict={'features:0': train_samples,
                                                      'targets:0': train_labels_onehot})

            for i in range(total_batch):
                batch_x = train_samples[i*batch_size:(i+1)*batch_size]
                batch_y = train_labels_onehot[i*batch_size:(i+1)*batch_size]
                _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,'targets:0': batch_y})
                avg_cost += c
        
            train_acc = sess.run('accuracy:0', feed_dict={'features:0': train_samples,
                                                      'targets:0': train_labels_onehot})
            test_acc,test_loss = sess.run(['accuracy:0','cost:0'], feed_dict={'features:0': test_samples,
                                                      'targets:0': test_labels_onehot})

            # Derive the distribution P(F1)
            Gibbs_all_f1 = Gibbs_pdf(activation1)
            Gibbs_all_f2 = Gibbs_pdf(activation2)
            Gibbs_all_f3 = Gibbs_pdf(activation3)
            Gibbs_all_fo = Gibbs_pdf(activationo)

            train_acc_vector[k,epoch] = train_acc
            test_acc_vector[k,epoch] = test_acc
            MIFX_vector[k,epoch] = [cal_MIFX(Gibbs_all_f1), cal_MIFX(Gibbs_all_f2), cal_MIFX(Gibbs_all_f3),cal_MIFX(Gibbs_all_fo)]
            MIFY_vector[k,epoch] = [cal_MIFY(Gibbs_all_f1,train_labels_onehot), cal_MIFY(Gibbs_all_f2,train_labels_onehot), cal_MIFY(Gibbs_all_f3,train_labels_onehot),cal_MIFY(Gibbs_all_fo,train_labels_onehot)]
            MIFXO_vector[k,epoch] = [MIFX_vector[k,epoch,0]-MIFY_vector[k,epoch,0],MIFX_vector[k,epoch,1]-MIFY_vector[k,epoch,1],MIFX_vector[k,epoch,2]-MIFY_vector[k,epoch,2],MIFX_vector[k,epoch,3]-MIFY_vector[k,epoch,3]]
            kld_vector[k,epoch] = avg_cost/(i + 1)
            kld_vector2[k,epoch] = test_loss

            print('MI(F1,X),  MI(F2,X),  MI(F3,X), and MI(FY,X)  are (%.2f, %.2f, %.2f, %.2f)'%(MIFX_vector[k,epoch,0],MIFX_vector[k,epoch,1],MIFX_vector[k,epoch,2],MIFX_vector[k,epoch,3]))
            print('MI(F1,Y),  MI(F2,Y),  MI(F3,X), and MI(FY,Y)  are (%.2f, %.2f, %.2f, %.2f)'%(MIFY_vector[k,epoch,0],MIFY_vector[k,epoch,1],MIFY_vector[k,epoch,2],MIFY_vector[k,epoch,3]))
            print('MI(F1,XO), MI(F2,XO), MI(F3,X), and MI(FY,XO) are (%.2f, %.2f, %.2f, %.2f)'%(MIFXO_vector[k,epoch,0],MIFXO_vector[k,epoch,1],MIFXO_vector[k,epoch,2],MIFXO_vector[k,epoch,3]))
            print("Nsamples: %2d,Epoch: %03d | AvgCost: %.3f/%.3f Train/Valid ACC: %.3f/%.3f" % (k+1,epoch + 1, avg_cost/(i + 1),test_loss,train_acc, test_acc))

        if activation_func == 'relu':
            np.savez('MLP_IT_FMNIST_%03d_%03d_%03d_relu.npz'%(n_hidden_1,n_hidden_2,n_hidden_3),train_acc=train_acc_vector,test_acc=test_acc_vector,kld=kld_vector,MIFX=MIFX_vector,MIFY=MIFY_vector,MIFXO=MIFXO_vector)
        elif activation_func == 'sigmoid':
            np.savez('MLP_IT_FMNIST_%03d_%03d_%03d_sigmoid.npz'%(n_hidden_1,n_hidden_2,n_hidden_3),train_acc=train_acc_vector,test_acc=test_acc_vector,kld=kld_vector, MIFX=MIFX_vector,MIFY=MIFY_vector,MIFXO=MIFXO_vector)
        else:
            np.savez('MLP_IT_FMNIST_%03d_%03d_%03d_tanh.npz'%(n_hidden_1,n_hidden_2,n_hidden_3),train_acc=train_acc_vector,test_acc=test_acc_vector,kld=kld_vector, MIFX=MIFX_vector,MIFY=MIFY_vector,MIFXO=MIFXO_vector)
    


