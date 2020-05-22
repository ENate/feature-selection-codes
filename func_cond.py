import os
import csv
import autograd.numpy as np
from autograd import elementwise_grad
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .grad_jcb_func import bs_ws_function
from .py_func_loss import func_all_params_concatenate


def softmax(a):
    expa = np.exp(a)
    return expa / expa.sum(axis=1, keepdims=True)


def activation_func(values_vec):
    """Mse based MLP loss function sigmoid."""
    eva_vectors = np.exp(- values_vec)
    sig_valued = 1 / (1 + eva_vectors)
    return np.array(sig_valued)


def activation_func2(values_vec):
    """Adapted sigmoid function for logistic."""
    return 0.5 * (np.tanh(values_vec / 2.) + 1)


def func_cond_compute_2(ws, bs, lambda_2, choose_flag, kwarg_dicts):
    xx = kwarg_dicts['xtr']
    Nx = xx.shape[0]
    """Call function to elementwise function for partial derivative of parameter."""
    # loss_func = func_eltwise_grad(ws, bs, kwarg_dicts)
    # Compute the autograd derivative of the function
    elt_grad0 = elementwise_grad(func_eltwise_grad, 0)
    elt_grad1 = elementwise_grad(func_eltwise_grad, 1)
    # Make a matrix of ws_0 and bs_0
    ws_bs_mat0 = np.r_[ws[0], bs[0]]
    # print('The shape of w[0] is:', ws[0].shape)
    # make a copy of the list of matrices
    cond_sum_val1 = 0.0
    cond_vector1 = np.zeros((ws_bs_mat0.shape[0], 1))
    cond_vector2 = np.zeros((ws_bs_mat0.shape[0], 1))
    if choose_flag != 1:
        for k_th in range(ws_bs_mat0.shape[0]):
            current_mat = ws_bs_mat0[k_th, :]
            grp_norm = np.sqrt(np.sum(np.square(current_mat)))
            # if grp_norm < lambda_2:
            # print('The grouped norm is: ', grp_norm)
            if k_th < ws[0].shape[0]:
                for j_th in range(ws_bs_mat0.shape[1]):
                    # for j_th in ws_bs_mat0.shape[1]:
                    # make a temp copy of the 0-the index of the matric list:
                    cpy_ws = ws.copy()
                    w0_cpy = cpy_ws[0].copy()
                    w0_cpy[k_th, j_th] = 0.0
                    cpy_ws[0] = w0_cpy.copy()
                    # set weight at (k, j) = 0 after copying to new matrix
                    # maybe set the local parameter to 0!
                    cond_val1 = elt_grad0(cpy_ws, bs, kwarg_dicts)[0][k_th, j_th]
                    cond_sum_val1 += (2*cond_val1) ** 2
            else:    
                for j2_th in range(ws_bs_mat0.shape[1]):
                    cpy_bs = bs.copy()
                    b0_cpy = cpy_bs[0].copy()
                    b0_cpy[0, j2_th] = 0.0
                    cpy_bs[0] = b0_cpy.copy()
                    cond_val1 = elt_grad1(ws, cpy_bs, kwarg_dicts)[0][0, j2_th]
                    # b0_cpy = 0.
                    cond_sum_val1 += (2*cond_val1)**2
            # else:
                    # cond_sum_val1 = 0.0
            cond_vector1[k_th, 0] = np.sqrt(cond_sum_val1)
            cond_sum_val1 = 0.0
        cond_sum_val1 = 0.0
        # check whether parameters at zero should remain zero
        
    else:
        cond_values_sum = 0.0
        for i_th in range(ws_bs_mat0.shape[0]):
            for d_th in range(ws_bs_mat0.shape[1]):
                elt_grad1 = elementwise_grad(func_eltwise_grad, 1)
                if ws_bs_mat0[i_th, d_th] == 0:
                    cond_val2 = 0.0
                else:
                    if i_th < ws_bs_mat0.shape[0] - 1:
                        ws_cpy = ws.copy()
                        ws0_cpy = ws_cpy[0].copy()
                        ws0_cpy[i_th, d_th] = 0.0
                        ws_cpy[0] = ws0_cpy.copy()
                        cond_val2 = (elt_grad0(ws_cpy, bs, kwarg_dicts)[0][i_th, d_th])**2
                    else:
                        bs_cpy = bs.copy()
                        bs0_cpy = bs_cpy[0].copy()
                        bs0_cpy[0, d_th] = 0.0
                        bs_cpy[0] = bs0_cpy.copy()
                        cond_val2 = (elt_grad1(ws, bs_cpy, kwarg_dicts)[0][0, d_th])**2
                cond_values_sum += cond_val2
            cond_vector1[i_th, 0] = np.sqrt(cond_values_sum)
            cond_values_sum = 0.0
        # cond_values_sum = 0.0
    return cond_vector1, ws, bs


def func_compute_cond(weights, lambda_3, kwargs_dict):
    """ Organize weights before processing """
    sess_values = kwargs_dict['sess']
    # weights equals sess_values.run(weights_tf)
    choose_flag = kwargs_dict['choose_flag']
    if choose_flag != 1:
        # regressor variables
        weight_sizes, weight_shapes = kwargs_dict['sizes'], kwargs_dict['shapes']
    else:
        weight_sizes, weight_shapes = kwargs_dict['wb_sizes'], kwargs_dict['wb_shapes']
    # A call to bs_ws_function to format the parameter vector
    ws_parms, bs_parms = bs_ws_function(weights, weight_shapes, weight_sizes)
    # A call to func_elementwise for partial derivative and compute
    cond_vec, ws, bs = func_cond_compute_2(ws_parms, bs_parms, lambda_3, choose_flag, kwargs_dict)
    # print(cond_vec)
    # joint first index matrices from ws and bs
    wb_0 = np.r_[ws_parms[0], bs_parms[0]]
    # apply condition to matrix connecting input -> hidden layer 1
    ws_matrix, ws_bs_in1_hid1 = func_set_mat0(wb_0, cond_vec, lambda_3, choose_flag)
    # separate them after setting corresponding matrices to zero
    ws_parms[0] = ws_matrix[0:-1, :].copy()
    bs_parms[0] = ws_matrix[-1:, :].copy()
    
    # return new ws_ and bs_ list of parameters
    # Either return formatter all_params
    new_all_params = func_all_params_concatenate(ws_parms, bs_parms)
    # Or part of all_params: connecting input -> hidden layer 1
    return new_all_params, ws_bs_in1_hid1, cond_vec


def func_set_mat0(ws_matrxi_0, condvec, lambda_3, choose_flag):
    """Check condition and set grouped parameters to zero satisfying condition."""
    # pass ws and bs matrices
    ws_matrxi_0 = np.array(ws_matrxi_0)
    condvec = np.array(condvec)
    # print(condvec)
    #if choose_flag == 1:
    # condvec = cond_vec2.copy()
    #else:
    #    condvec = cond_vec.copy()

    for ind in range(ws_matrxi_0.shape[0]):
        # for ind2 in range(ws_matrxi_0.shape[1]):
        
        if condvec[ind] < lambda_3:
            ws_matrxi_0[ind, :] = 0.0
    ws_parms_0 = ws_matrxi_0[0:-1, :]
    bs_parms_0 = ws_matrxi_0[-1:,:]
    # reshape part of all_params connecting in->hidden layer 1
    ws_reshaped = np.reshape(ws_parms_0, ws_parms_0.shape[0] * ws_parms_0.shape[1])
    bs_reshape = np.reshape(bs_parms_0, bs_parms_0.shape[0] * bs_parms_0.shape[1])
    in_2_hid1_mat = np.r_[ws_reshaped, bs_reshape]
    return ws_matrxi_0, in_2_hid1_mat


def func_eltwise_grad(ws_classify, bs_classify, kwarg_dict):
    """Compute condition in this function."""
    choose_flag = kwarg_dict['choose_flag']
    if choose_flag == 1:
        x_values, y_label, hidden = kwarg_dict['xtr'], kwarg_dict['ytr'], kwarg_dict['hidden']
        N = x_values.shape[0]
        y_hat = x_values
        for k_idx in range(len(hidden)):
            y_hat = activation_func(np.dot(y_hat, ws_classify[k_idx]) + bs_classify[k_idx])
        y_hat = ((np.dot(y_hat, ws_classify[-1]) + bs_classify[-1]))
        label_probs = (y_hat * y_label + (1 - y_hat) * (1 - y_label))
        # N is the y_hat of shape
        N = 1/(y_hat.shape[0])
        cost_prob = - N * (np.sum(y_label * np.log(y_hat) + (1 - y_label) * (np.log(1- y_hat))))
        # cost_prob =  - N * np.sum(np.log(label_probs))
    else:
        x_values, y_label, hidden = kwarg_dict['xtr'], kwarg_dict['ytr'], kwarg_dict['hidden']
        # p_sizes and p_shapes equals kwarg_dict['sizes'], kwarg_dict['shapes']
        y_hat = x_values
        for k_idx in range(len(hidden)):
            y_hat = activation_func(np.dot(y_hat, ws_classify[k_idx]) + bs_classify[k_idx])
        y_hat = np.dot(y_hat, ws_classify[-1]) + bs_classify[-1]
        # y_hat equals np.squeeze(y_hat)
        r_loss = y_label - y_hat
        resdue = np.square(r_loss)
        cost_prob = np.sum(np.sum(resdue, 1))
    return cost_prob
