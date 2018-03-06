#encoding=utf-8
import numpy as np
import os, sys
from sklearn.datasets import make_moons,load_digits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from activation_functions import relu,sigmoid,relu_backward,sigmoid_backward,tanh,tanh_backward,softmax,one_hot
from mini_batches import random_mini_batches
from nn_with_momentum import initialize_velocity,update_parameter_with_momentum
import pickle

BasePath = sys.path[0]

def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: -- a list contains the number of neuron
    :return:
    parameter -- a dictionary contains "Wl" and "bl"
    """
    np.random.seed(1)
    L = len(layer_dims)
    parameter = dict()
    for l in range(1,L):
        parameter["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameter["b" + str(l)] = np.zeros((layer_dims[l],1))
    return parameter

def linear_forward(A_prev, W, b):
    """

    :param A_prev: -- the output of previous layer, shape of (number of features of previous layer, number of examples)
    :param param: -- the parameter of this layer, a dictionary contains "Wl" and "bl"
                        "Wl" shape of (layer_dims[l],layer_dims[l-1])
    :return:
    Zl -- the linear output of this layer  ,shape of (number of features of this layer, number of examples)
    linear_cache -- a dictionary conatins A_prev, W, and b
    """
    Z = np.dot(W, A_prev) + b
    linear_cache = dict()
    linear_cache["A_prev"] = A_prev
    linear_cache["W"] = W
    linear_cache["b"] = b
    return Z, linear_cache

def activation_forward(A_prev, W, b, activation_way):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation_way: -- a text string indicate the way we activate this layer, "sigmoid","relu",...
    :return:
    A -- the activation output of this layer
    cache -- a dictionary contains "linear_cache' and "activation_cache"
    """
    cache = dict()
    if activation_way == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation_way == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation_way == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation_way == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    cache["linear_cache"] = linear_cache
    cache["activation_cache"] = activation_cache

    return A, cache

def L_layer_forward(X, param):
    """

    :param X: -- the input data, shape of (number of features, number of examples)
    :param layer_dims: -- contains [X.shape[0], n1,n2,...,nL]
    :return:
    AL -- the output of last layer, also the prediction of Y
    caches -- a list contains many cache, each cache is a dictionary that contains "linear_cache" and "activation_cache"
    """
    np.random.seed(1)
    # param = initialize_parameters_deep(layer_dims)
    L = len(param)//2
    caches = list()
    A = X
    for l in range(1,L):
        A, cache = activation_forward(A, param["W"+str(l)], param["b"+str(l)], "tanh")
        caches.append(cache)
    AL, cache = activation_forward(A, param["W"+str(L)], param["b"+str(L)], "softmax")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    """

    :param AL: -- the activation output of layer L, also prediction of Y
    :param Y: -- the true label of X data, shape of (number features of layer L, number of examples)
    :return:
    cost -- the total loss over m examples, also call J
    """
    m = Y.shape[1]
    cost = -(np.dot(Y,np.log(AL).T) + np.dot(1-Y,np.log(1-AL).T))/m

    return cost

def C_classes_compute_cost(AL, Y):
    """

    :param AL: -- the activation output of layer L, also prediction of Y, shape (number of classes, number of examples)
    :param Y: -- the true label of X data, shape of (number features of layer L, number of examples)
    :return:
    cost --
    """
    m = Y.shape[1]
    Loss = - Y * np.log(AL)
    cost = np.sum(Loss) / m

    return cost

def linear_backward(dZ, linear_cache):
    """

    :param dZ: -- the linear output gradient with respect to J, shape (nl, # of examples)
    :param linear_cache:-- a dictionary that contains "A_prev", "W" and "b"
    :return:
    dW -- the gradient of W this layer respect to the cost J
    db -- the gradient of b this layer respect to the cost J
    dA_prev -- the gradient of activation previous layer respect to the cost J
    """
    m = dZ.shape[1]
    A_prev = linear_cache["A_prev"]
    W = linear_cache["W"]
    b = linear_cache["b"]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims =True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def activation_backward(dA, cache, activation_way):
    """

    :param dA: Gradient of the activation output this layer with respect to cost J, shape(nl, #)
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dA_prev -- the gradient of activation previous layer respect to the cost J
    dW -- the gradient of W this layer respect to the cost J
    db -- the gradient of b this layer respect to the cost J
    """
    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]

    if activation_way == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev,dW,db = linear_backward(dZ, linear_cache)
    elif activation_way == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation_way == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_layer_backward(AL, caches, Y):
    """

    :param AL: the activation output of layer L, shape(nl, #)
    :param caches: -- a list [cache0,...,cacheL-1], each cache is a dictionary contains "linear_cache" and "activation_cache"
    :param Y: -- the true label of data X
    :return:
    grads -- a dictionary contains each layer "dWl" and "dbl"
    """
    L = len(caches)
    m = Y.shape[1]
    grads = dict()
    dAL = -Y/AL + (1-Y)/(1-AL)
    dA_prev,dWL,dbL = activation_backward(dAL, caches[-1], "sigmoid")
    grads["dW"+str(L)] = dWL
    grads["db"+str(L)] = dbL

    for l in reversed(range(L-1)):
        dA_prev,dW,db = activation_backward(dA_prev, caches[l], "tanh")
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db

    return grads

def C_classes_backward(AL, caches, Y):
    """

    :param AL:
    :param caches:
    :param Y:
    :return:
    """
    L = len(caches)
    m = Y.shape[1]
    grads = dict()
    dZL = AL - Y

    dA_prev, dWL, dbL = linear_backward(dZL, caches[-1]["linear_cache"])
    grads["dW" + str(L)] = dWL
    grads["db" + str(L)] = dbL

    for l in reversed(range(L-1)):
        dA_prev,dW,db = activation_backward(dA_prev, caches[l], "tanh")
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db

    return grads

def update_parameter(parameter, grads, learning_rate = 0.0075):
    """

    :param parameter: -- a dictionary contains each layer's parameter ,i.e."W1", "b1", "W2", "b2",...
    :param grads: -- a dictionary contains each layer's gradient respect to cost ,i.e. "dW1","db1",...
    :return:
    parameter -- updated parameter
    """
    L = len(parameter)//2
    for l in range(1,L+1):
        parameter["W" + str(l)] = parameter["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameter["b" + str(l)] = parameter["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameter

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075,beta=0.9, num_iteration=400, cost_plot = False):
    """

    :param X: -- input data, numpy array, shape (number of features, number of examples)
    :param layer_dims: -- a list contains number of neuron each layer
    :return:
    param -- parameters that have been optimized
    """
    np.random.seed(1)
    costs = list()
    mini_batches = random_mini_batches(X, Y, mini_batch_size=64, seed=1)
    parameter = initialize_parameters_deep(layer_dims)
    v_momentum = initialize_velocity(parameter)
    for i in range(num_iteration):
        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch
            AL, caches = L_layer_forward(mini_batch_X, parameter)

            # cost = compute_cost(AL, Y)
            cost = C_classes_compute_cost(AL, mini_batch_Y)
            costs.append(cost)

            # grads = L_layer_backward(AL, caches, Y)
            grads = C_classes_backward(AL, caches, mini_batch_Y)

            # parameter = update_parameter(parameter, grads, learning_rate)
            parameter, v_momentum = update_parameter_with_momentum(parameter, grads, v_momentum, beta=beta, learning_rate=learning_rate)
            # print("Cost after iteration {}, {}th mini-batch: {}".format(i,j, np.squeeze(cost)))
            # costs.append(cost)
        # if i%100 == 0:
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        costs.append(cost)
    if cost_plot:
        plt.plot(np.squeeze(costs))
        plt.title("learning_rate = " + str(learning_rate))
        plt.xlabel("Number of iteration")
        plt.ylabel("Cost of loss")
        plt.show()

    return parameter

def prepare_data(filePath):
    """

    :param filePath:
    :return:
    X -- data of features
    """
    images = list()
    file_list = os.listdir(filePath)
    for file in file_list:
        child = os.path.join('%s%s' % (filePath,file))
        image = mpimg.imread(child)
        image = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
        images.append(image)
    X = np.column_stack(tuple(images))

    return X

def get_data_from_sklearn(n_class):
    """

    :return:
    X -- input data, numpy array, shape (number of features, number of examples)
    Y -- the true label of X
    """
    # X, Y = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=None)
    digits = load_digits(n_class=n_class, return_X_y=True)
    X_all, Y_all = digits
    X_all = X_all.T
    m = X_all.shape[1]
    # print(m)
    permutation = list(np.random.permutation(m))
    X_all = X_all[:,permutation]
    Y_all = Y_all.reshape(1, len(Y_all))
    Y_all = one_hot(Y_all, n_class)
    Y_all = Y_all[:,permutation]
    X_train = X_all[:,:1000]
    Y_train = Y_all[:,:1000]
    X_test = X_all[:,1000:]
    Y_test = Y_all[:,1000:]

    return X_train, Y_train, X_test, Y_test

def get_label(filePath):
    """

    :param filePath:
    :return:
    Y -- a numpy array, shape of (number of classes, number of examples)
    """
    Y = list()
    with open(filePath,'r') as f:
        line = f.readline()
        while(line):
            label_list = line.split()
            for label in label_list:
                label = label.strip()
                if label != "":
                    Y.append(int(label))
            line = f.readline()
    Y = np.array(Y).reshape(1, len(Y))
    return Y

def predict(X, parameter):
    """

    :param X: -- the input data
    :param Y: -- the true label of data X
    :param parameter: -- the W and b that have been optimized
    :return:
    Y_hat -- the prediction of Y
    accuracy -- the prediction accuracy
    """
    Y_predict ,caches = L_layer_forward(X, parameter)
    Y_hat = Y_predict >= 0.5
    # accuracy = np.mean(Y==Y_hat)
    return Y_hat

def draw_boundary(X, Y, param):
    """

    :param X:
    :param Y:
    :param param:
    :return:
    """
    # x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    # y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max),
    #                      np.arange(y_min, y_max))

    # here "model" is your model's prediction (classification) function
    # Z = model(np.c_[xx.ravel(), yy.ravel()])
    # Z = predict(np.c_[xx.ravel(), yy.ravel()], param)
    Y_hat = predict(X, param)
    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    p1 = plt.subplot(121)
    p1.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Paired)
    p1.legend("True distribution")
    p2 = plt.subplot(122)
    p2.scatter(X[0, :], X[1, :], c=Y_hat, cmap=plt.cm.Paired)
    p2.legend("The prediction")
    # plt.axis('off')
    # Plot also the training points
    plt.show()

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data_from_sklearn(n_class=10)
    # X_train = prepare_data(BasePath + '/imgs_train/')
    # X_train = X_train/255
    # Y_train = get_label(BasePath + '/Y_label.txt')
    layer_dims = [X_train.shape[0], 40, 30, 20, 10]
    # # keep_prob = [0.5,0.8]
    param = L_layer_model(X_train, Y_train, layer_dims, learning_rate=0.05,num_iteration=500, cost_plot=True)
    # with open('param_ccl_with_momentum.pickle','wb') as f:
    #     pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
    # with open('param_ccl_with_momentum.pickle','rb') as f:
    #     param = pickle.load(f)
    # Y_hat, accu = predict(X_train, Y_train, param)
    # print(accu)
    # print(Y_hat)