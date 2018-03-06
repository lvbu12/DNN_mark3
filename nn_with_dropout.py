#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from build_nn import get_data_from_sklearn, predict
from build_nn import linear_backward,activation_forward,activation_backward,initialize_parameters_deep,update_parameter,C_classes_compute_cost
import pickle

def L_layer_forward_with_dropout(X, parameter, keep_prob):
    """

    :param X: -- the input data, shape of (number of features, number of examples)
    :param parameter: -- python dictionary contains "Wl" and "bl"
    :param keep_prob: -- probability of keeping a neuron active during drop-out
    :return:
    AL -- the activation output of layer L, also the prediction of Y
    caches -- a list of cache, each cache contains "linear_cache" and "activation_cache"
    """
    L = len(parameter)//2
    caches = list()
    A = X
    D_cache = dict()
    for l in range(1,L):
        A, cache = activation_forward(A, parameter["W"+str(l)], parameter["b"+str(l)], "tanh")
        # print(A.shape)
        caches.append(cache)
        D = np.random.rand(A.shape[0], A.shape[1])
        # print(D.shape)
        D = D < keep_prob[l-1]
        # print("after " + str(D.shape))
        D_cache["D"+str(l)] = D
        A = A * D
        A = A / keep_prob[l-1]

    AL, cache = activation_forward(A, parameter["W" + str(L)], parameter["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches, D_cache

def L_layer_backward_with_dropout(AL, caches, D_cache, Y, keep_prob):
    """

    :param AL: -- the activation output of layer L, shape(nL, #)
    :param caches: -- a list [cache0,...,cacheL-1], each cache is a dictionary contains "linear_cache" and "activation_cache"
    :param D_cache: a dictionary contains "D1", "D2",...,"DL-1"
    :param Y: -- the true label of data X
    :param keep_prob: -- a list of each layer's probability of keeping a neuron activation during drop-out
    :return:
    grads -- a dictionary contains each layer "dWl" and "dbl"
    """
    L = len(caches)
    m = Y.shape[1]
    grads = dict()
    # dAL = -Y / AL + (1 - Y) / (1 - AL)
    dZL = AL - Y
    # dA_prev, dWL, dbL = activation_backward(dAL, caches[-1], "sigmoid")
    dA_prev, dWL, dbL = linear_backward(dZL, caches[-1]["linear_cache"])
    grads["dW" + str(L)] = dWL
    grads["db" + str(L)] = dbL
    for l in reversed(range(L-1)):
        dA_prev = dA_prev*D_cache["D"+str(l+1)]
        dA_prev /= keep_prob[l]
        dA_prev,dW,db = activation_backward(dA_prev, caches[l], "tanh")
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db

    return grads

def L_layer_model(X, Y, layer_dims, keep_prob, learning_rate=0.0075, num_iteration=400, cost_plot = False):
    """

    :param X: -- input data, numpy array, shape (number of features, number of examples)
    :param layer_dims: -- a list contains number of neuron each layer
    :return:
    param -- parameters that have been optimized
    """
    np.random.seed(1)
    costs = list()
    parameter = initialize_parameters_deep(layer_dims)
    for i in range(num_iteration):
        # AL, caches = L_layer_forward(X, parameter)
        AL, caches, D_cache = L_layer_forward_with_dropout(X, parameter, keep_prob)
        cost = C_classes_compute_cost(AL, Y)
        # cost = compute_cost_with_regularization(AL, Y, parameter, lambd=0.7)
        costs.append(cost)
        # grads = L_layer_backward(AL, caches, Y, lambd=0.7)
        # grads = L_layer_backward_with_regularization(AL, Y, caches, lambd=0.7)
        grads = L_layer_backward_with_dropout(AL, caches, D_cache, Y, keep_prob)
        parameter = update_parameter(parameter, grads, learning_rate)

        if i%100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    if cost_plot:
        plt.plot(np.squeeze(costs))
        plt.title("learning_rate = " + str(learning_rate))
        plt.xlabel("Number of iteration")
        plt.ylabel("Cost of loss")
        plt.show()

    return parameter

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data_from_sklearn(n_class=10)
    # layer_dims = [X_train.shape[0], 40, 30, 20, 10]
    # keep_prob = [0.5, 0.7, 0.8]
    # param = L_layer_model(X_train, Y_train, layer_dims, keep_prob, learning_rate=0.05, num_iteration=10000, cost_plot=True)
    # with open('param_with_d.pickle','wb') as f:
    #     pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
    with open('param_with_d.pickle','rb') as f:
        param = pickle.load(f)
    Y_hat, accu = predict(X_test, Y_test, param)
    print(accu)
