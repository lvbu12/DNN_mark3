#_*_ coding=utf-8 _*_
import numpy as np
from array_change import dictionary_to_vector,vector_to_dictionary,grads_to_vector
from build_nn import L_layer_forward,compute_cost,initialize_parameters_deep,prepare_data,BasePath,get_label,L_layer_backward

def check_gradient(parameter, X, Y,layer_dims, epsilon = 1e-7):
    """

    :param parameter: -- a dictionary contains "Wl" and "bl"
    :param grads: -- a dictionary contains each layer "dWl" and "dbl"
    :param X: -- the input data, shape of (number of features, number of examples)
    :param Y: -- the true label of X data, shape of (number features of layer L, number of examples)
    :param epsilon:
    :return:
    difference -- difference between the approximated gradient and backward propagation gradient
    """
    AL, caches = L_layer_forward(X, parameter)
    grads = L_layer_backward(AL, caches, Y)

    param_vec = dictionary_to_vector(parameter)
    grads_vec = grads_to_vector(grads)

    num_param = param_vec.shape[0]
    J_plus = np.zeros((num_param, 1))
    J_minus = np.zeros((num_param, 1))
    gradapprox = np.zeros((num_param, 1))

    for i in range(num_param):
        theta_plus = param_vec
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        param = vector_to_dictionary(theta_plus, layer_dims)
        AL, caches = L_layer_forward(X, param)
        J_plus[i] = compute_cost(AL, Y)

        theta_minus = param_vec
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        param = vector_to_dictionary(theta_minus, layer_dims)
        AL, caches = L_layer_forward(X, param)
        J_minus[i] = compute_cost(AL, Y)

        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

    numerator = np.linalg.norm(grads_vec - gradapprox)
    denominator = np.linalg.norm(grads_vec) + np.linalg.norm(gradapprox)
    difference = numerator/denominator

    if difference > 2e-7:
        print("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print("Your backward propagation works well! difference = " + str(difference))
    return difference

if __name__ =="__main__":
    X = prepare_data(BasePath + '/imgs_train/')
    Y = get_label(BasePath + '/Y_label.txt')
    layer_dims = [X.shape[0], 4, 3, 1]
    parameter = initialize_parameters_deep(layer_dims)
    diff = check_gradient(parameter, X, Y, layer_dims)
    print(diff)


