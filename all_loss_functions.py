import numpy as np
import tensorflow as tf
# structures l2 penalty


# def func_structured_l2pen(ws0_matrix, bs0matrix):
def func_structured_l2pen(ps, sizes_w, shapes_w):
    # ws0_matrix = tf.reshape(ps[0:sizes_w[0]], shape=shapes_w[0])
    # bs0matrix = tf.reshape(ps[sizes_w[0]:], shape=shapes_w[1])
    # combined_wb_matrices = tf.concat([ws0_matrix, bs0matrix], axis=0)
    shaped_new = np.int(sizes_w[0]) + np.int(sizes_w[1])
    idx1 = np.int(shapes_w[0][0]) + np.int(shapes_w[1][0])
    lasso_p = ps[shaped_new:]
    l2_p = ps[0:shaped_new]
    combined_wb_matrices = tf.reshape(l2_p, shape=(idx1, shapes_w[0][1]))
    sq_params = tf.square(combined_wb_matrices)
    colwise_add = tf.reduce_sum(sq_params, axis= 1)
    sqrt_rowise_p = tf.sqrt(colwise_add)
    gen_sum_params = tf.reduce_sum(sqrt_rowise_p)
    # all_reg_0 = tf.reduce_sum(tf.abs(lasso_p))
    all_reg_0 = tf.reduce_sum(tf.abs(ps))
    return gen_sum_params, all_reg_0, l2_p, ps#, lasso_p#, ps


# classifier l2
def func_structured_l2pen_classifier(ws0_matrix, bs0matrix):
    combined_wb_matrices = tf.concat([ws0_matrix, bs0matrix], axis=0)
    sq_params = tf.square(combined_wb_matrices)
    colwise_add = tf.reduce_sum(sq_params, axis= 1)
    sqrt_rowise_p = tf.sqrt(colwise_add)
    gen_sum_params = tf.reduce_sum(sqrt_rowise_p)
    return gen_sum_params
# MSE regression with any of the penalties


def func_mse_loss(n_new, nn_new, **kwargs):
    """ Determine the loss function """
    shapes_new, sizes_new, neurons_cnt_new = kwargs['shapes'], kwargs['sizes'], kwargs['neurons_cnt']
    # placeholder variables (we have m data points)
    neurons_cnt = kwargs['neurons_cnt']
    initializer, activation = kwargs['initializer'], kwargs['activation']
    x_place_hold = tf.placeholder(tf.float64, shape=[None, n_new])
    y = tf.placeholder(tf.float64, shape=[None, 1])
    p = tf.Variable(initializer([neurons_cnt_new], dtype=tf.float64))
    x_params = tf.split(p, sizes_new)
    for i_index in range(len(x_params)):
        x_params[i_index] = tf.reshape(x_params[i_index], shapes_new[i_index])
    weights = x_params[0:][::2]
    biases = x_params[1:][::2]
    y_hat = x_place_hold
    for i in range(len(nn_new)):
        y_hat = activation(tf.matmul(y_hat, weights[i]) + biases[i])
    y_hat = tf.matmul(y_hat, weights[-1]) + biases[-1]
    y_hat_flat_values = tf.squeeze(y_hat)
    return y_hat, y_hat_flat_values, y, p, x_place_hold


def func_mse_l2(n_input_pts, p_init, ll_hid, kwargs_d):
    m_shape, m_sizes, m_neurons_cnt = kwargs_d['shapes'], kwargs_d['sizes'], kwargs_d['neurons_cnt']
    initializer, activation = kwargs_d['initializer'], kwargs_d['activation']
    x_holder = tf.placeholder(tf.float64, shape=[None, n_input_pts])
    y_holder = tf.placeholder(tf.float64, shape=[None, 1])
    x_init_parameters = tf.split(p_init, m_sizes)
    for k_idx in range(len(x_init_parameters)):
        x_init_parameters[k_idx] = tf.reshape(x_init_parameters[k_idx], m_shape[k_idx])
    l2_weights = x_init_parameters[0:][::2]
    l2_biases = x_init_parameters[1:][::2]
    l2_model = x_holder
    for l_idx in range(len(ll_hid)):
        l2_model = activation(tf.matmul(l2_model, l2_weights[l_idx]) + l2_biases[l_idx])
    l2_model = tf.matmul(l2_model, l2_weights[-1]) + l2_biases[-1]
    l2_model_flat = tf.squeeze(l2_model)
    r = y_holder - l2_model
    # l2_norm_val = func_structured_l2pen(l2_weights[0], l2_biases[0])
    l2_norm_val, allreg0, l2_p, lasso_p = func_structured_l2pen(p_init, m_sizes, m_shape)
    return l2_model, l2_model_flat, y_holder, x_holder, r, l2_norm_val, allreg0, l2_p, lasso_p


# try to apply method to classification for cross entropy loss
def func_cross_entropy_loss(wb_sizes_classif, params, kwargs1):
    nclassif, hidden, activation = kwargs1['n'], kwargs1['hidden'], kwargs1['activation'] 
    initializer, neurons_cnt_classif, wb_shapes = kwargs1['initializer'], kwargs1['neurons_cnt'], kwargs1['wb_shapes']
    # n labels is for 2 # number of output classes or labels
    n_class_labels = kwargs1['nclasses']
    xclassif = tf.placeholder(tf.float64, shape=[None, nclassif])
    yclassif = tf.placeholder(tf.float64, shape = [None, n_class_labels])
    classif_tensors = tf.split(params, wb_sizes_classif, 0)
    for i in range(len(classif_tensors)):
        classif_tensors[i] = tf.reshape(classif_tensors[i], wb_shapes[i])
    ws_classif = classif_tensors[0:][::2]
    bs_classif = classif_tensors[1:][::2]
    y_hat_classif = xclassif
    for i in range(len(hidden)):
        y_hat_classif = tf.nn.sigmoid(tf.matmul(y_hat_classif, ws_classif[i]) + bs_classif[i])
        # y_hat_classif = tf.nn.sigmoid()
    y_hat_classif = tf.matmul(y_hat_classif, ws_classif[-1]) + bs_classif[-1]
    #################################################################################
    structuredl2pen = func_structured_l2pen_classifier(ws_classif[0], bs_classif[0])
    ####################################################################################
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_classif, labels=yclassif)) # + regparam
    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat_classif, labels=yclassif))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat_classif, labels=tf.cast(yclassif,tf.float64)))
    return loss, xclassif, yclassif, y_hat_classif, structuredl2pen
