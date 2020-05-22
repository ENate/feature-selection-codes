import autograd.numpy as np
import tensorflow as tf

def sig_activation(y_hat_vec_mat):
    sig_vector = 1 / (1 + np.exp(- y_hat_vec_mat))
    return sig_vector
#

def func_hess(all_parameters, inputs, y, kwargs1):
    #
    wb_sizes_x, wb_shapes = kwargs1['wb_sizes'], kwargs1['wb_shapes']
    x_wb_size_new = []
    sum_wb_sizes = 0
    wb_sizes_array = np.asarray(wb_sizes_x) 
    # Format the size entries using number python package
    for k in range(len(wb_sizes_array)):
        sum_wb_sizes += wb_sizes_array[k]
        x_wb_size_new.append(sum_wb_sizes)

    #print(x_wb_size_new)
    #print(wb_shapes)
    split_params_lst = []
    split_params = np.split(all_parameters, x_wb_size_new)
    for i in range(len(split_params) - 1):
        split_params_lst.append(np.reshape(split_params[i], wb_shapes[i]))
    ws2_classify = split_params_lst[0:][::2]
    bs2_classify = split_params_lst[1:][::2]
    # print(ws2_classify)
    hidden = kwargs1['hidden']
    y_hat = inputs
    for k_idx in range(len(hidden)):
        y_hat = sig_activation(np.dot(y_hat, ws2_classify[k_idx]) + bs2_classify[k_idx])
    y_hat = np.dot(y_hat, ws2_classify[-1]) + bs2_classify[-1]
    r = y - y_hat
    y_loss = np.sum(np.sum(np.square(r), 1))

    return y_loss

def func_predict(opt_p, x_in, y_out, wb_shapes, wb_sizes_x1, hidden1):
    # wb_sizes_x1, wb_shapes = kwargs1['wb_sizes'], kwargs1['wb_shapes']
    x_wb_size_new1 = []
    sum_wb_sizes1 = 0
    wb_sizes_array1 = np.asarray(wb_sizes_x1)
    # Format the size entries using number python package
    for k in range(len(wb_sizes_array1)):
        sum_wb_sizes1 += wb_sizes_array1[k]
        x_wb_size_new1.append(sum_wb_sizes1)

    print(x_wb_size_new1)
    print(wb_shapes)
    split_params_lst1 = []
    split_params1 = np.split(opt_p, x_wb_size_new1)
    print(len(split_params1[1]))
    for i in range(len(split_params1) - 1):
        split_params_lst1.append(np.reshape(split_params1[i], wb_shapes[i]))
    ws2_classify1 = split_params_lst1[0:][::2]
    bs2_classify1 = split_params_lst1[1:][::2]
    # print(ws2_classify)
    # hidden1= kwargs1['hidden']
    y_hat1 = x_in
    for k_idx in range(len(hidden1)):
        y_hat1 = sig_activation(np.dot(y_hat1, ws2_classify1[k_idx]) + bs2_classify1[k_idx])
    y_hat1 = np.dot(y_hat1, ws2_classify1[-1]) + bs2_classify1[-1]
    return y_hat1

