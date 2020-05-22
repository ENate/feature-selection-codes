import numpy as np
import tensorflow as tf


def predclassif(wb_sizes_classif, xydat, hidden, params, activation, wb_shapes, nclasses):
    # placeholder variables (we have m data points)
    # n labels is for 2 # number of output classes or labels
    xtest1, ytest1 = xydat[0], xydat[1]
    nclassif = xtest1.shape[1]
    xclassif = tf.placeholder(tf.float64, shape=[None, nclassif])
    # labels = tf.placeholder(tf.int64, shape=[None, ])
    labels = tf.placeholder(tf.float64, shape=[None, nclasses])
    # labels_one_hot = tf.one_hot(labels, 2)
    feed_dict2 = {xclassif: xtest1, labels: ytest1}
    classif_tensors = tf.split(params, wb_sizes_classif, 0)

    for i in range(len(classif_tensors)):
        classif_tensors[i] = tf.reshape(classif_tensors[i], wb_shapes[i])
    ws_classif = classif_tensors[0:][::2]
    bs_classif = classif_tensors[1:][::2]
    y_hat_classif_logits = xclassif
    for i in range(len(hidden)):
        model0 = tf.matmul(y_hat_classif_logits, ws_classif[i])
        y_hat_classif_logits = tf.nn.sigmoid(model0 + bs_classif[i])
    y_hat_classif_logits = tf.nn.sigmoid(tf.matmul(y_hat_classif_logits, ws_classif[-1]) + bs_classif[-1])
    correct_prediction = tf.equal(tf.argmax(y_hat_classif_logits, 1), tf.argmax(labels, 1)) 
    return correct_prediction, feed_dict2, y_hat_classif_logits


def func_pred(nn, p, sizes, shapes, activation, xtest):
    # placeholder variables (we have m data points)
    xtest00 = np.asanyarray(xtest)
    biases = np.ones((xtest00.shape[0], 1))
    xtr = np.c_[xtest00, biases]

    n = xtr.shape[1]
    nm = xtr.shape[0]

    x = tf.placeholder(tf.float64, shape=[nm, n])
    parms = tf.split(p, sizes, 0)
    for i in range(len(parms)):
        parms[i] = tf.reshape(parms[i], shapes[i])
    Ws = parms[0:][::2]
    bs = parms[1:][::2]

    y_hat = x
    for i in range(len(nn)):
        y_hat = tf.nn.sigmoid(tf.matmul(y_hat, Ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, Ws[-1])+bs[-1]
    y_hat_flat = tf.squeeze(y_hat)
    feed_dict = {x: xtr}
    sess = tf.Session()
    predy = sess.run(y_hat_flat, feed_dict)
    sess.close()
    return predy


def func_pred_new(n_inputs, nn_0, p, **kwargs):
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
