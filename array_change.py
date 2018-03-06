# encoding = utf-8
import numpy as np
from build_nn import initialize_parameters_deep


def dictionary_to_vector(parameter):
    """

    :param parameter: -- a dictionary contains "Wl", "bl"
    :return:
    param_vec -- a vector transferred by parameter
    """
    L = len(parameter)//2
    param_vec = list()
    for l in range(1,L+1):
        tmp_vec_W = parameter["W"+str(l)].flatten()
        tmp_vec_b = parameter["b" + str(l)].flatten()
        param_vec = np.hstack((param_vec, tmp_vec_W, tmp_vec_b))
    param_vec = np.array(param_vec).reshape(len(param_vec),1)
    return param_vec

def grads_to_vector(grads):
    """

    :param parameter: -- a dictionary contains "Wl", "bl"
    :return:
    param_vec -- a vector transferred by parameter
    """
    L = len(grads)//2
    param_vec = list()
    for l in range(1,L+1):
        tmp_vec_W = grads["dW"+str(l)].flatten()
        tmp_vec_b = grads["db" + str(l)].flatten()
        param_vec = np.hstack((param_vec, tmp_vec_W, tmp_vec_b))
    param_vec = np.array(param_vec).reshape(len(param_vec),1)
    return param_vec

def vector_to_dictionary(param_vec, layer_dims):
    """

    :param param_vec: -- a vector contains "Wi" and "bi", flatted parameter
    :param layer_dims: -- a list contains [n0,n1,n2,...,nL]
    :return:
    :parameter -- a dictionary contains "Wl" and "bl"
    """
    L = len(layer_dims)
    parameter = dict()
    index_W, index_b = 0, 0
    for l in range(1, L):
        parameter["W"+str(l)] = param_vec[index_W:index_W + layer_dims[l]*layer_dims[l-1]].reshape(layer_dims[l], layer_dims[l-1])
        index_b = index_W + layer_dims[l]*layer_dims[l-1]
        parameter["b"+str(l)] = param_vec[index_b:index_b + layer_dims[l]].reshape(layer_dims[l],1)
        index_W = index_b + layer_dims[l]
    return parameter

if __name__ == "__main__":
    layer_dims = [5,4,3,1]
    param = initialize_parameters_deep(layer_dims)
    print(param)
    param_vec = dictionary_to_vector(param)
    parameter = vector_to_dictionary(param_vec, layer_dims)
    print(parameter)
