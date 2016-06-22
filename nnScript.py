import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt  
from numpy import ndarray
from scipy.special import expit
import datetime
import pickle

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
    
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.zeros((data.shape[0],1))
    
    new_biasH = np.ones(len(data)) # Add a dimension for bias with value 1

    data = np.column_stack([data, new_biasH])
    
    net_hidden = np.dot(data ,w1.transpose())  # net_p at hidden layer 50000*50
    
    out_hidden = sigmoid(net_hidden)# output at hidden layer 50000*50
    
    new_biasO = np.ones(len(out_hidden)) # Add a dimension for bias with value 1

    out_hidden = np.column_stack([out_hidden, new_biasO])
    
    net_out = np.dot(out_hidden,w2.transpose()) # net_l at output layer
    
    out_l = sigmoid(net_out) 
    
    for x in range(out_l.shape[0]):
        maxInd = np.argmax(out_l[x])
        labels[x] = maxInd
    
    return labels.transpose()
    
def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
#Load the training data and stack into a single array. Also create true labels for each digit
    train = ndarray([784,],int) 
    #train_label = ndarray([10,],int)
    label_train = list()

    for i in range(10):
        m = mat.get('train'+str(i))
        t = np.array(m)
        label_train.extend([i]*len(t))
        train = np.vstack((train,t))
        
    label_train = np.asarray(label_train) #convert the label list into array
    
    # Feature selection for training and validation data starts
    z2 =  np.all(train == train[0,:], axis = 0)
    z2 = np.invert(z2)
    train = train[:,z2]
    
    # Feature selection ends
    
    #Normalize the vector to have values between 0 and 1    
    train = train[1:len(train)]/255

    #Pick a reasonable size for validation data
    validation_size = 10000

    #Divide training data into two parts: training and validation data

    tot_size = range(train.shape[0])
    perm = np.random.permutation(tot_size) #random permutation for splitting data
    validation_data= train[perm[0:validation_size],:]
    train_data= train[perm[validation_size :],:]

    #Divide training labels into two parts: training and validation labels
    validation_label = label_train[perm[0:validation_size]]
    train_label = label_train[perm[validation_size:]]
    
    #Load the testing data into a single array

    test_data = ndarray([784,],int) 
    #test_label = ndarray([10,],int)
    label_test = list()
    for i in range(10):
        m = mat.get('test'+str(i))
        t = np.array(m)
        label_test.extend([i]*len(t))
        test_data = np.vstack((test_data,t))

    test_data = test_data[1:len(test_data)]
    
    test_label = np.asarray(label_test) #convert the label list into array
    test_label = test_label.transpose()
    
    # Feature selection for test data
    test_data = test_data[:,z2]
    
  
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    

    # Feed Forward Propagation
    
    yl = np.zeros((len(training_data),10))
    for x in range(len(training_data)):
        yl[x][training_label[x]]=1
        
    training_label = yl
    
    #print(training_label.shape)
    
    new_biasH = np.ones(len(training_data)) # Add a dimension for bias with value 1

    training_data = np.column_stack([training_data, new_biasH])
    
    net_hidden = np.dot(training_data ,w1.transpose())  # net_p at hidden layer 50000*50
    
    out_hidden = sigmoid(net_hidden)# output at hidden layer 50000*50
    
    new_biasO = np.ones(len(out_hidden)) # Add a dimension for bias with value 1

    out_hidden1 = np.column_stack([out_hidden, new_biasO]) # 50000*51
    
    net_out = np.dot(out_hidden1,w2.transpose()) # net_l at output layer
    
    out_l = sigmoid(net_out)  # 50000*10
    
    # Feed forward propagation ends
    
    # Backward Feed Propagation Starts
    
    # error_delta_param1 = np.dot((out_l) ,(1 - out_l.transpose())) # 50000 * 50000
    #error_delta_param1 = (1 - out_l)* out_l
    
    #error_delta_o = error_delta_param1 * (training_label - out_l)  # 50000*10
    #error_delta_o = (training_label - out_l)  # 50000*10
    error_delta_o = (out_l - training_label)
    
    grad_w2 = np.dot(out_hidden1.transpose(), error_delta_o) # 51 * 10
    
    #print(grad_w2)
    # Parameters for graient of W1
    
    #w1_par1 = np.dot(out_hidden, (1 - out_hidden.transpose())) # 50000*50000
    w1_par1 = (1 - out_hidden1) * out_hidden1
    w1_par2 = np.dot(error_delta_o , w2) # 50000 *51

    w1_par3 = w1_par1 * w1_par2 # 50000 * 51

    grad_w1 = np.dot(training_data.transpose(), w1_par3) # 785 * 51
    
    grad_w1 = grad_w1[:,0:n_hidden]
    #print(grad_w1)

    # Error function calculation Starts

    err_par1 = 0.5 *((training_label - out_l)**2)
    err_par1 = (np.sum(err_par1))
    
    divisor = len(training_data)
    
    err_function = err_par1/divisor
    
    # Error function calculation Ends
    
    # Regularization Calculation Starts
    
    reg_par1 = np.sum(w1*w1)
    reg_par2 = np.sum(w2*w2)
    
    reg_par3 = (lambdaval/(2*divisor))*(reg_par1 + reg_par2)
    
    reg_function = err_function + reg_par3

    #Update the gradient functions
    
    grad_w1 = (grad_w1.transpose() + (lambdaval*w1))/divisor # 50*785
    
    grad_w2 = (grad_w2.transpose() + (lambdaval*w2))/divisor # 10*51
    
    obj_val = reg_function

    #print('\n' + str(obj_val))
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)



"""**************Neural Network Script Starts here********************************"""
current_time = datetime.datetime.now().time()
print('\n Start time: ' + str(current_time))
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

print('\n Train Data Shape' + str(train_data.shape))
print('\n validation_data Shape' + str(validation_data.shape))
print('\n Test Data Shape' + str(test_data.shape))

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

#print('\ncalling obj function')

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Test Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

current_time = datetime.datetime.now().time()
print('\n End time: ' + str(current_time))

pickle.dump( [n_hidden, w1, w2, lambdaval], open( "params.pickle", "wb" ) )

