  
'''
---------------------------------- START OF DOCUMENT ----------------------------------
'''

'''
my_submission.py file that is the developed code for Assignment #2 in Unit IFN680 at Queensland University of Technology in Brisbane during Semester 2, 2019.
Due date for submission is the 27th of October @ 11.59pm
This assignment is submitted by:
    Donghan Jiang       n10075615
    Lin Dong            n10359613
'''

# modules imported 
import numpy as np
from numpy import array
import keras.backend as K
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Input, Lambda
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda,Conv2D,MaxPooling2D,BatchNormalization
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt 
'''
------------------------------------------------------------------------------
List of functions in this .py document:
+ pre_processing(x_train, y_train, x_test, y_test)
+ reshape_convert_input_data(input_data)
+ create_cnn(input_shape)
+ create_improved_cnn(input_shape)
+ euclidean_distance(vects)
+ eucl_dist_output_shape(shapes)
+ contrastive_loss_function(y_true, y_pred)
+ create_pairs_set(x, digit_indices, test_index)
+ compute_accuracy(y_true, y_pred)
+ accuracy(y_true, y_pred)
+ create_siamese()
+ get_siamese_pair(set_lable,set_image,dataset)
+ test_and_plot(model,tr_pairs, tr_target, te_pairs1, te_target1,epochs)
------------------- END OF LIST OF FUNCTIONS ---------------------------------
'''
# data preprocessing 
def pre_processing(x_train,y_train,x_test,y_test):
    '''
    -args: 
        x_train, y_train, x_test, y_test
        
    -return: splitted data
        image_set1_train, 
        image_set1_test, 
        label_set1_train, 
        label_set1_test, 
        image_set2,
        label_set2,
        image_set3,
        label_set3
    '''
    
    # concatenate data into one set
    image_all = np.concatenate((x_train, x_test), axis=0);
    label_all = np.concatenate((y_train, y_test), axis=0);
    
    #show the class name to help viewing 
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    
    # print(image_all.shape) 
    
    #dataset which needs to train and test
    dataset_train_and_test = [0,1,2,4,5,9]
    
    # create emmpty set for data processing
    image_set1 = []
    label_set1 = []
    image_set2 = []
    label_set2 = []
    
    # allocate data into two sets based on the requirements
    for i in range(len(image_all)):
        if label_all[i] in dataset_train_and_test:
            image_set1.append(image_all[i])
            label_set1.append(label_all[i])
        else:
            image_set2.append(image_all[i])
            label_set2.append(label_all[i])
        
    # change the sets to arrays
    image_set1 = array(image_set1)
    label_set1 = array(label_set1)
    image_set2 = array(image_set2)
    label_set2 = array(label_set2)
    
    # reshape the datasets
    image_set1 = image_set1.reshape(((image_set1.shape[0], 28, 28, 1)))
    image_set2 = image_set2.reshape(((image_set2.shape[0], 28, 28, 1)))
    
    # convert from integers to floats
    image_set1 = image_set1.astype('float32')
    image_set2 = image_set2.astype('float32')
    
    # normalize to range 0-1
    image_set1 = image_set1 / 255.0
    image_set2 = image_set2 / 255.0
    
    # split train and test dataset
    image_set1_train, image_set1_test, label_set1_train, label_set1_test = train_test_split(image_set1, label_set1, test_size =0.2, random_state = 42)
    
    # concatenate testset 
    image_set3 = np.concatenate((image_set1_test, image_set2), axis=0)
    label_set3 =np.concatenate((label_set1_test, label_set2), axis=0)
    
    return image_set1_train, image_set1_test, label_set1_train, label_set1_test, image_set2,label_set2,image_set3,label_set3

def euclidean_distance(vects):
    '''
    This function is used to calculate Euclidean distance
    
    -args: 
        vests
        
    -return:
        the distance of the pairs
    '''
    
    x, y = vects
    
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# return the Euclidean distance shape
    
def eucl_dist_output_shape(shapes):
    '''
    This function is used to get Euclidean distance output shape
    
    -args: 
        shape
        
    -return:
        (shape1[0], 1)
    '''
    
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_cnn(input_shape):
    '''
    This function is used to create base network to be shared (eq. to feature extraction).
    
    -args: 
        input_shape
        
    -return:
        model
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32,(3,3),activation='relu')(input)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_improved_cnn(input_shape):
    '''
    This function is used to create improved network to be shared (eq. to feature extraction).
    
    -args: 
        input_shape
        
    -return:
        model
    '''
    #add convmaxpooling2d
    #add maxpooling2d
    #add dropout
    input = Input(shape=input_shape)
    x = Conv2D(32,(3,3),activation='relu')(input)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
    
    
def compute_accuracy(y_true, y_pred):
    '''
    For evaluating the prediction accuracy of the model.
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    -returns:
        Accuracy
    '''
    
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    
    '''
    Computes classification accuracy with a fixed threshold on distances.
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''
    
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def create_pairs(x, digit_indices):
    
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.  
    -args: 
        x, digit_indices
        
    -return:
        np.array(pairs), np.array(labels)
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(len(digit_indices))]) - 1
    for d in range(len(digit_indices)):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, len(digit_indices))
            dn = (d + inc) % len(digit_indices)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

# create training+test positive and negative pairs
def create_siamese(): 
    '''
    create siamese network
    -args: 
        
    -return:
        siamese model
    '''
    input_shape = image_set1_train.shape[1:]      
    # network definition
    base_network = create_cnn(input_shape)
    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    return model

# create training+test positive and negative pairs by improved cnn
def create_improved_siamese(): 
    '''
    create siamese network based on improved cnn
    -args: 
        
    -return:
        siamese model
    '''
    input_shape = image_set1_train.shape[1:]      
    # network definition
    base_network = create_improved_cnn(input_shape)
    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    return model

def get_siamese_pair(set_lable,set_image,dataset):
    '''
    create siamese pair 
    -args: 
        et_lable,set_image,dataset
    -return:
        tr_pairs,tr_target
    '''
    digit_indices = [np.where(set_lable == i)[0] for i in dataset]
    tr_pairs, tr_target = create_pairs(set_image, digit_indices)
    return tr_pairs,tr_target

def test_and_plot(model,tr_pairs, tr_target, te_pairs1, te_target1,epochs):
    '''
    print and plot the accuaracy
    -args: 
        model,tr_pairs, tr_target, te_pairs1, te_target1,epochs
    -return:
        
    '''
    
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_target,
              batch_size=128,
              epochs = epochs,
              validation_data=([te_pairs1[:, 0], te_pairs1[:, 1]], te_target1))
  # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_target, y_pred)
    y_pred = model.predict([te_pairs1[:, 0], te_pairs1[:, 1]])
    te_acc = compute_accuracy(te_target1, y_pred)  
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print(history.history.keys())
    #Plot Accuracy for training data and testing data
    plt.plot(history.history['accuracy'], 'blue')
    plt.plot(history.history['val_accuracy'], 'orange')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    #Plot loss results for training data and testing data
    plt.plot(history.history['loss'], 'blue')
    plt.plot(history.history['val_loss'], 'orange')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

 # main application
 # Split differnt datasets for training and testing
dataset_train_and_test = [0,1,2,4,5,9] 
dataset_test_only = [3,6,7,8]
dataset_all = [0,1,2,3,4,5,6,7,8,9]
# import fashion mnist dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
#data_preprocessing
'''
    dataset name with 1 means this data comes from dataset_train_and_test
    dataset name with 2 means this data comes from dataset_test_only 
    dataset name with 3 means this data comes from dataset_all
'''
image_set1_train, image_set1_test, label_set1_train, label_set1_test,image_set2,label_set2,image_set3,label_set3=pre_processing(x_train,y_train,x_test,y_test)
#get pairs
tr_pairs, tr_target = get_siamese_pair(label_set1_train,image_set1_train,dataset_train_and_test)
te_pairs1, te_target1 = get_siamese_pair(label_set1_test,image_set1_test,dataset_train_and_test)
te_pairs2, te_target2 = get_siamese_pair(label_set2,image_set2,dataset_test_only)
te_pairs3, te_target3 = get_siamese_pair(label_set3,image_set3,dataset_all)
# create siamese network
model = create_siamese()
# create siamese network based on improved cnn
model_improved = create_improved_siamese()

# try 3 different numbers of epochs

epochs = [5,10,15]

for i in epochs:
    print("epochs: " , i)
    print("Training on 80% and validation on 20% dataset with 6 classes")
    test_and_plot(model,tr_pairs, tr_target, te_pairs1, te_target1,i)
    print("Training on 80% dataset with 6 classes and validation on all the dataset with 4 classes" )
    test_and_plot(model,tr_pairs, tr_target, te_pairs2, te_target2,i)
    print("Training on 80% dataset with 6 classes and validation on all the dataset with 4 classes plus 20% dataset with 6 classes")
    test_and_plot(model,tr_pairs, tr_target, te_pairs3, te_target3,i)
    
print("Use improved CNN network")

for i in epochs:
    print("epochs: " , i)
    print("Training on 80% and validation on 20% dataset with 6 classes")
    test_and_plot(model_improved,tr_pairs, tr_target, te_pairs1, te_target1,i)
    print("Training on 80% dataset with 6 classes and validation on all the dataset with 4 classes" )
    test_and_plot(model_improved,tr_pairs, tr_target, te_pairs2, te_target2,i)
    print("Training on 80% dataset with 6 classes and validation on all the dataset with 4 classes plus 20% dataset with 6 classes")
    test_and_plot(model_improved,tr_pairs, tr_target, te_pairs3, te_target3,i)