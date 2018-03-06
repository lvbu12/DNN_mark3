#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from build_nn import initialize_parameters_deep,get_data_from_sklearn,L_layer_forward,C_classes_compute_cost,C_classes_backward,update_parameter,predict
from mini_batches import random_mini_batches
from nn_with_momentum import initialize_velocity, update_parameter_with_momentum
import pickle

def initialize_adam(parameter):
    """

    :param parameter: -- python dictionary contains "Wl" and "bl"
    :return:
    v -- python dictionary contains exponentially weighted average of the gradient
    s -- python dictionary contains exponentially weighted of the squared gradient
    """
    L = len(parameter)//2
    v_adam = dict()
    s_adam = dict()
    for l in range(1, L+1):
        v_adam["dW" + str(l)] = np.zeros((parameter["W" + str(l)].shape))
        v_adam["db" + str(l)] = np.zeros((parameter["b" + str(l)].shape))
        s_adam["dW" + str(l)] = np.zeros((parameter["W" + str(l)].shape))
        s_adam["db" + str(l)] = np.zeros((parameter["b" + str(l)].shape))

    return v_adam, s_adam

def update_parameter_with_adam(parameter, grads, v_adam, s_adam, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """

    :param parameter: -- python dictionary contains "Wl" and "bl"
    :param grads: -- python dictionary contains "dWl" and "dbl"
    :param v: -- adam variable, python dictionary, moving average of the first gradient
    :param s:-- Adam variable,python dictionary, moving average of the squared gradient,
    :param t:
    :param learning_rate:
    :param beta1:-- exponential decay hyper-parameter for the first moment estimates
    :param beta2:-- exponential decay hyper-parameter for the second moment estimates
    :param epsilon:-- hyper-parameter preventing division by zero
    :return:
    parameter -- python dictionary contains your updated parameter
    v -- Adam variable, python dictionary, moving average of the first gradient
    s -- Adam variable,python dictionary, moving average of the squared gradient
    """
    L = len(parameter)//2
    v_corrected = dict()
    s_corrected = dict()
    # Perform Adam update on all parameters
    for l in range(1, L+1):
        # Moving average of the gradients.
        v_adam["dW" + str(l)] = beta1 * v_adam["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v_adam["db" + str(l)] = beta1 * v_adam["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # Compute bias-corrected first moment estimate.
        v_corrected["dW" + str(l)] = v_adam["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v_adam["db" + str(l)] / (1 - beta1**t)
        # Moving average of the squared gradients.
        s_adam["dW" + str(l)] = beta2 * s_adam["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s_adam["db" + str(l)] = beta2 * s_adam["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        # Compute bias-corrected second raw moment estimate.
        s_corrected["dW" + str(l)] = s_adam["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s_adam["db" + str(l)] / (1 - beta2 ** t)
        # Update parameters.
        parameter["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameter["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameter, v_adam, s_adam

def model(X, Y, layer_dims, optimizer, learning_rate=0.0075, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True, plot_cost=False):
    """

    :param X:
    :param Y:
    :param layer_dims:
    :param optimizer:
    :param learning_rate:
    :param mini_batch_size:
    :param beta:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param num_epochs:
    :param print_cost:
    :return:
    parameter -- python dictionary contains your updated paramter
    """
    costs = []
    t = 0
    seed = 1
    # Initialize parameters
    parameter = initialize_parameters_deep(layer_dims)
    # Initialize the optimizer
    if optimizer == "gd":
        pass                # no initialization required for gradient descent
    elif optimizer == "momentum":
        v_momentum = initialize_velocity(parameter)
    elif optimizer == "adam":
        v_adam, s_adam = initialize_adam(parameter)
    # Optimization loop
    for i in range(num_epochs):
        # Define the random minibatches.
        seed = seed + 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size=mini_batch_size, seed=seed)
        for mini_batch in mini_batches:
            t += 1
            (mini_batch_X, mini_batch_Y) = mini_batch

            # AL, caches = forward_propagation(mini_batch_X, parameter)
            AL, caches = L_layer_forward(mini_batch_X, parameter)

            cost = C_classes_compute_cost(AL, mini_batch_Y)
            costs.append(cost)

            # grads = backward_propagation(mini_batch_X, mini_batch_Y, caches)
            grads = C_classes_backward(AL, caches, mini_batch_Y)
            if optimizer == "gd":
                # parameter = update_parameter_with_gd(parameter, grads, learning_rate)
                parameter = update_parameter(parameter, grads, learning_rate)
            elif optimizer == "momentum":
                parameter, v_momentum = update_parameter_with_momentum(parameter, grads, v_momentum, beta, learning_rate)
            elif optimizer == "adam":
                parameter, v_adam, s_adam = update_parameter_with_adam(parameter, grads, v_adam, s_adam, t, learning_rate, beta1, beta2, epsilon)

        # if print_cost and i % 100 == 0:
        print("Cost after epoch %i: %f" % (i, cost))
        costs.append(cost)
    if plot_cost:
        plt.plot(costs)
        plt.xlabel("epochs")
        plt.ylabel("Cost")
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
    return parameter

def predict(X,Y, parameter):
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
    accuracy = np.mean(Y==Y_hat)
    return Y_hat, accuracy

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data_from_sklearn(n_class=10)
    # layer_dims = [X_train.shape[0], 40, 30, 20, 10]
    # param = model(X_train, Y_train, layer_dims, optimizer="adam", learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=500, print_cost=True, plot_cost=True)
    # with open('param_adam.pickle','wb') as f:
    #     pickle.dump(param,f,pickle.HIGHEST_PROTOCOL)
    with open('param_adam.pickle','rb') as f:
        param = pickle.load(f)
    Y_hat,accu = predict(X_test, Y_test, param)
    print(accu)