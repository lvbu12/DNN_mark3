#encoding=utf-8
import numpy as np

def relu(Z):
    """

    :param Z: -- the linear output in this layer
    :return:
    A -- the activation output in this layer
    activation_cache -- a dictionary contains Z and A
    """
    [m, n] = Z.shape
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if Z[i][j] < 0:
                A[i][j] = 0
            else:
                A[i][j] = Z[i][j]
    activation_cache = dict()
    activation_cache["Z"] = Z
    activation_cache["A"] = A
    return A, activation_cache

def relu_backward(dA, activation_cache):
    """

    :param dA: -- Gradient of activation of this layer respect to cost, shape(nl, # of examples)
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dZ -- the gradient of linear output of this layer, same shape as dA
    """
    [m,n] = dA.shape
    dZ = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if dA[i][j] < 0:
                dZ[i][j] = 0
            else:
                dZ[i][j] = dA[i][j]
    return dZ

def leaky_relu(Z):
    """

    :param Z: -- the linear output of this layer
    :return:
    A -- the activation output of this layer
    activation_cache
    """
    [m,n] = Z.shape
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if Z[i][j] < 0:
                A[i][j] = 0.01 * Z[i][j]
            else:
                A[i][j] = Z[i][j]
    activation_cache = dict()
    activation_cache["A"] = A
    activation_cache["Z"] = Z
    return A, activation_cache

def leaky_relu_backward(dA, activation_cache):
    """

    :param dA: -- Gradient of activation of this layer respect to cost, shape(nl, # of examples)
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dZ -- the gradient of linear output of this layer, same shape as dA
    """
    [m,n] = dA.shape
    dZ = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if dA[i][j] < 0:
                dZ[i][j] = 0.01 * dA[i][j]
            else:
                dZ[i][j] = dA[i][j]
    return dZ


def sigmoid(Z):
    """

    :param Z: -- the linear output in this layer
    :return:
    A -- the activation output of this layer
    activation_cache -- a dictionary contains "A" and "Z"
    """
    A = 1/(1+np.exp(-Z))
    activation_cache = dict()
    activation_cache["A"] = A
    activation_cache["Z"] = Z
    return A, activation_cache

def sigmoid_backward(dA, activation_cache):
    """

    :param dA: -- Gradient of activation of this layer respect to cost, shape(nl, # of examples)
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dZ -- the gradient of linear output of this layer, same shape as dA
    """
    A = activation_cache["A"]
    dZ = dA * A * (1-A)
    return dZ

def tanh(Z):
    """

    :param Z: -- the output of linear forward of this layer
    :return:
    A -- the activation output of this layer
    actiavtion_cache -- a dictionary contains "A' and "Z"
    """
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    activation_cache = dict()
    activation_cache["A"] = A
    activation_cache["Z"] = Z
    return A, activation_cache

def tanh_backward(dA, activation_cache):
    """

    :param dA: -- Gradient of activation of this layer respect to cost, shape(nl, # of examples)
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dZ -- the gradient of linear output of this layer, same shape as dA
    """
    A = activation_cache["A"]
    dZ = dA * (1 - A*A)
    return dZ

def softmax(Z):
    """

    :param Z:-- the linear output of layer L,
    :return:
    A -- the activation output of the final layer, A = softmax(Z)
    """
    [m,n] = Z.shape
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            A[i][j] = np.exp(Z[i][j])/np.sum(np.exp(Z[:,j]))
    activation_cache = dict()
    activation_cache["A"] = A
    activation_cache["Z"] = Z
    return A, activation_cache

def softmax_backward(dA, activation_cache):
    """

    :param dA: -- Gradient with respect to AL
    :param activation_cache: -- a dictionary contains "A" and "Z"
    :return:
    dZ -- Gradient of cost with respect to ZL
    """
    pass

def one_hot(Y, C):
    """

    :param Y: -- shape of (1, number of examples)
    :param C: -- number of classes of Y
    :return:
    Y_oh -- shape of (C, number of examples)
    """
    m = Y.shape[1]
    Y_oh = np.zeros((C, m))
    for i in range(m):
        Y_oh[Y[0,i],i] = 1
    return Y_oh