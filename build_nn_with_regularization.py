#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from build_nn import C_classes_compute_cost,C_classes_backward,initialize_parameters_deep,L_layer_forward,update_parameter,get_data_from_sklearn,predict
import pickle

def compute_cost_with_regularization(AL, Y, parameters, lambd = 0.7):
    """

    :param AL: -- the output of layer L, also the prediction of Y
    :param Y: -- the true label of data X
    :param parameters: -- a dictionary contains "Wl" and "bl"
    :param lambd: -- hyper-parameter of L2 regularization
    :return:
    cost -- total loss with regularization
    """
    L2_regularization_cost = 0
    L = len(parameters)//2
    m = Y.shape[1]
    for i in range(1,L+1):
        L2_regularization_cost += lambd/2*np.sum(np.square(parameters["W"+str(i)]))/m
    cross_entropy_cost = C_classes_compute_cost(AL, Y)
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

def L_layer_backward_with_regularization(AL, Y, caches, lambd=0.7):
    """

    :param X: -- input data set, shape (input size, number of examples)
    :param Y: -- true labels of X
    :param caches: -- a list of cache, each cache is a dictionary contains "linear_cache" and "activatin_cache"
    :param lambd: -- regularization hyper-parameter ,scalar
    :return:
    grads -- a dictionary contains "dW" and "db"
    """
    grads = C_classes_backward(AL, caches, Y)
    m = Y.shape[1]
    L = len(caches)
    for l in range(1,L+1):
        grads["dW"+str(l)] = grads["dW"+str(l)] + lambd*caches[l-1]["linear_cache"]["W"]/m
    return grads

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iteration=400, cost_plot = False, lambd = 0.7):
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
        AL, caches = L_layer_forward(X, parameter)
        # AL, caches, D_cache = L_layer_forward_with_dropout(X, parameter, keep_prob)
        # cost = compute_cost(AL, Y)
        cost = compute_cost_with_regularization(AL, Y, parameter, lambd=0.7)
        costs.append(cost)
        # grads = L_layer_backward(AL, caches, Y, lambd=0.7)
        grads = L_layer_backward_with_regularization(AL, Y, caches, lambd=0.7)
        # grads = L_layer_backward_with_dropout(AL, caches, D_cache, Y, keep_prob)
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
    # param = L_layer_model(X_train, Y_train, layer_dims, learning_rate=0.05, num_iteration=10000, cost_plot=True)
    # param = L_layer_model(X_train, Y_train, layer_dims, learning_rate=0.05, num_iteration=10000, cost_plot=True, lambd=0.7)
    # with open('param_with_r.pickle','wb') as f:
    #     pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
    with open('param_with_r.pickle','rb') as f:
        param = pickle.load(f)
    Y_hat, accu = predict(X_test, Y_test, param)
    print(accu)