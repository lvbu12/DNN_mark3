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
