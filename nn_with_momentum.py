#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def initialize_velocity(parameter):
    """

    :param parameter: -- a dictionary contains "Wl" and "bl"
    :return:
    v -- a dictionary contains the current velocity
            v['dW'+str(l)] = velocity of dWl
            v['db'+str(l)] = velocity of dbl
    """
    L = len(parameter)//2
    v_momentum = dict()
    # Initialize velocity
    for l in range(1, L+1):
        v_momentum["dW" + str(l)] = np.zeros((parameter["W"+str(l)].shape))
        v_momentum["db" + str(l)] = np.zeros((parameter["b"+str(l)].shape))
    return v_momentum

def update_parameter_with_momentum(parameter, grads, v_momentum, beta=0.9, learning_rate=0.0075):
    """

    :param parameter: -- python dictionary contains "Wl", "bl"
    :param grads: -- python dictionary contains "dWl","dbl"
    :param v: -- python dictionary contains the current velocity:
                v["dW"+str(l)] = dWl
                v["db"+str(l)] = dbl
    :param beta: -- the momentum hyper-parameter
    :param learning_rate:
    :return:
    parameter -- python dictionary contains your updated parameter
    v -- python dictionary contains your update velocity
    """
    L = len(parameter)//2
    for l in range(1, L+1):
        v_momentum["dW"+str(l)] = beta*v_momentum["dW"+str(l)]+(1-beta)*grads["dW"+str(l)]
        v_momentum["db"+str(l)] = beta*v_momentum["db"+str(l)]+(1-beta)*grads["db"+str(l)]

        parameter["W" + str(l)] -= learning_rate*v_momentum["dW" + str(l)]
        parameter["b" + str(l)] -= learning_rate*v_momentum["db" + str(l)]
    return parameter, v_momentum

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