import os
import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .drawoptstructure import paramreshape
from .deepNetImpl import NeuralNetwork
from .func_cond import func_compute_cond
from .py_func_loss import func_collect_allparams
from .py_lasso_l2 import model_l1_l2_func, model_main_func_l1


def load_solar_data(mdatfile, datmsg):
    """ This is the mat lab data function """
    mat_contents = sio.loadmat(mdatfile, struct_as_record=False)
    oct_struct = mat_contents[datmsg]
    if datmsg == 'testData':
        valdata = oct_struct[0, 0].xyvalues
        #valdata = (valdata - valdata.min(0)) / valdata.ptp(0)
        x_data = valdata[:, 0:-1]
        y_data = valdata[:, -1]
    
    else:
        # datmsg == 'solardatanorm':
        valdata = oct_struct[0, 0].values
        x_data = valdata[:, 0:-1]
        y_data = valdata[:, -1]
    x_train, x_test, y_train_set, y_test_set = train_test_split(x_data, y_data, test_size=0.20, shuffle=False)
    return x_train, x_test, y_train_set, y_test_set


def data_to_train0(xtrain0, ytr0, xt0):
    """ Format the data from the mat lab function """
    xtrain00 = np.asanyarray(xtrain0)
    biases = np.ones((xtrain00.shape[0], 1))
    xtr = np.c_[xtrain00, biases]
    # For the sake of testing
    xt00 = np.asanyarray(xt0)
    biases = np.ones((xt00.shape[0], 1))
    xt1 = np.c_[xt00, biases]
    ytr = ytr0[:, None]
    n = xtrain00.shape[1]
    Nm = xtr.shape[0]
    return Nm, n, xtrain00, xt00, ytr


def build_network_model(n, nn_hidden):
    # nn = [15]
    st = [n] + nn_hidden + [1]
    shapes = []
    for i in range(len(nn_hidden) + 1):
        shapes.append((st[i], st[i + 1]))
        shapes.append((1, st[i + 1]))
    sizes = [h * w for h, w in shapes]
    neurons_cnt = sum(sizes)
    return shapes, sizes, neurons_cnt, nn_hidden


def activation_func():
    activation = tf.nn.sigmoid
    return activation


def jacobian_mse(y, x_var, n_m):
    """compute the Jacbian matrix """
    stopgrads = tf.where(tf.math.equal(x_var, 0))
    loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(tf.float64, size=n_m), ]
    _, jacobian_mse = tf.while_loop(lambda i, _: i < nm,
        lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x_var)[0])), loop_vars)
        # lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x_var, unconnected_gradients='zero', stop_gradients=stopgrads)[0])), loop_vars)
    dxdt = tf.gradients(tf.reduce_sum(tf.abs(x_var)), x_var)[0]
    return jacobian.stack() , dxdt


def func_loss(n_new, nn_new, **kwargs):
    """ Determine the loss function """
    shapes_new, sizes_new, neurons_cnt_new = kwargs['shapes'], \
                                             kwargs['sizes'], kwargs['neurons_cnt']
    # placeholder variables (we have m data points)
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


def func_pred(n_inputs, nn_0, p, **kwargs):
    shapes_vec, sizes_vec = kwargs['shapes'], kwargs['sizes']
    activation = kwargs['activation']
    # placeholder variables (we have m data points)
    x_inputs = tf.placeholder(tf.float64, shape=[None, n_inputs])
    parameter_vector = tf.split(p, sizes_vec, 0)
    for i in range(len(parameter_vector)):
        parameter_vector[i] = tf.reshape(parameter_vector[i], shapes_vec[i])
    ws = parameter_vector[0:][::2]
    bs = parameter_vector[1:][::2]
    y_hat = x_inputs
    for i in range(len(nn_0)):
        y_hat = activation(tf.matmul(y_hat, ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, ws[-1]) + bs[-1]
    y_hat_flat_out = tf.squeeze(y_hat)
    return y_hat_flat_out, x_inputs


def train_classifier_sgd(x, y, loss, train_step, **kwargs3):
    """ classifier loss?"""
    step = 0
    batch_size = 5
    x_dat = kwargs3['xtrain']
    y_dat = kwargs3['ytrain']
    tf_train_labels = kwargs3['tf_train_labels']
    tf_train_dataset = kwargs3['tf_train_dataset']
    train_prediction = kwargs3['train_prediction']
    optimizer = kwargs3['optimizer']
    train_labels = y_dat
    train_dataset = x_dat
    feed_dict = {x: x_dat, y: y_dat}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calc initial loss
    current_loss = session.run(loss, feed_dict)
    while current_loss > 1e-10 and step < 400:
        step += 1
        # log(step, current_loss, session.run(params))
        session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a mini-batch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the
        # session where to feed the mini-batch.
        # The key of the dictionary is the
        # placeholder node of the graph to be fed
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    return current_loss


def model_main_func(nm_set_points, n_in, nn_1, opt_obj, **kwargs_values):
    hess_approx = False
    neurons_cnt_x = kwargs_values['neurons_cnt']
    all_sizes, all_shapes = kwargs_values['sizes'], kwargs_values['shapes']
    y_hat, y_hat_flat_x, y, p, x_in = func_loss(n, nn, **kwargs_values)
    x_trained = kwargs_values['xtr']
    y_trained = kwargs_values['ytr']
    all_reg_0 = tf.reduce_sum(tf.abs(p))
    r = y - y_hat
    lambda1 = 0.0001
    loss = tf.reduce_mean(tf.square(r)) + lambda1 * all_reg_0
    mu = tf.placeholder(tf.float64, shape=[1])
    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=tf.float64))
    save_params_p = tf.assign(p_store, p)
    restore_params_p = tf.assign(p, p_store)
    I_mat = tf.eye(neurons_cnt, dtype=tf.float64)
    I_mat_diag = tf.eye(neurons_cnt, dtype=tf.float64)

    if hess_approx:
        j1, dx_dt = jacobian_mse(y_hat, p, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        jtr = 2 * (tf.matmul(jt, r)) + lambda1 * tf.reshape(dx_dt, shape=(neurons_cnt, 1))
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss, p)[0]
        jtr = -tf.gradients(loss, p)[0] # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)

    dx = tf.squeeze(dx)
    lm = opt.apply_gradients([(-dx, p)])
    # p2 = p.assign(p + dx)
    sess_values = kwargs_values['sess']
    feed_dict = {x_in: xtr, y: ytr}
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss, feed_dict)
    zero0 = tf.constant(0., dtype=tf.float64)
    while feed_dict[mu] > 1e-10 and step < 300:
        p0 = sess_values.run(p)
        p_0_indices = np.where(p == 0)
        p0[p_0_indices] = 0.0
        step += 1
        sess.run(save_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(p_store)
        for _ in range(400):

            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(p)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0)
            p0[values_vec] = 0.0
            # print(sess.run(p0))
            new_loss = sess_values.run(loss, feed_dict)
            if new_loss < current_loss:
                mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    p_0_indices = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[p_0_indices] = 0.
                    print(len(mat_values))
                    p.assign(px)
                    print(p_0_indices)
                    i_cnt = 0
                    mat_values = []
                    mat_values = [px]
                else:
                    p.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                p.assign(p0)
                # sess_values.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break
    p_new = sess_values.run(restore_params_p)
    p22 = p_new.copy()
    p_new[abs(p_new) <= 1e-02] = 0.0
    p20 = p_new
    p20[p_0_indices] = 0.0
    print('ENDED ON STEP: ', ' FINAL LOSS:')
    print(step, current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print('p2:')
    print(p20)
    return p22, p20