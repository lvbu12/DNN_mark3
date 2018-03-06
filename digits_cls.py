#encoding=utf-8
import numpy as np
import pandas as pd
from activation_functions import one_hot
from build_nn import BasePath, L_layer_model,predict
import pickle

def get_data_from_kaggle(filePath):
    """

    :param filePath:
    :return:
    X -- input data, shape of (number of features, number of examples)
    Y -- the true label of X, shape of (number of class, number of examples)
    """
    my_frame = pd.read_csv(filePath)
    data = my_frame.values
    Y = data[:,0].T
    Y = Y.reshape(1,len(Y))
    Y = one_hot(Y, 10)
    X = data[:,1:].T
    return X, Y

def get_Y_label(Y_hat):
    """

    :param Y_hat: -- shape of (number of class, number of examples)
    :return:
    """
    [m,n] = Y_hat.shape
    Y_label = np.zeros((1,n))
    for i in range(n):
        Y_label[0,i] = np.argmax(Y_hat[:,i])
    return Y_label

if __name__ == "__main__":
    # X_train, Y_train = get_data_from_kaggle(BasePath + '/digits/train.csv')
    test_frame = pd.read_csv(BasePath + '/digits/test.csv')
    test_data = test_frame.values
    X_test = test_data.T
    X_test = X_test/255
    # X_train = X_train/255
    # layer_dims = [X_train.shape[0], 40,20,10]
    # param = L_layer_model(X_train, Y_train, layer_dims, learning_rate=0.05,beta=0.9, num_iteration=20, cost_plot = True)
    # with open('digits_cls.pickle','wb') as f:
    #     pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
    with open('digits_cls.pickle','rb') as f:
        param = pickle.load(f)
    Y_hat = predict(X_test, param)
    Y_label = get_Y_label(Y_hat).T

    m = len(Y_label)
    idx = np.array(list(range(1,m+1)),dtype=int)
    data_to_csv = np.column_stack((idx,Y_label))
    test_frame = pd.DataFrame(data_to_csv, columns=["ImageId","Label"])
    test_frame.to_csv("submission.csv", encoding="utf-8")
    # print(accu)

