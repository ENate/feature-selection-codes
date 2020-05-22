"""Script to train Rosenbrock function MLP approximation.

A simple usage example:
python classifStructuredL2.py -N=7 -m=20000 -hidden=[16,12,8] -opt=lm \
                           -kwargs={'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':10} \
                           -out=log1.txt
python classifStructuredL2.py -N=7 -m=20000 -hidden=[16,12,8] -opt=sgd \
                           -kwargs={'learning_rate':1e-3} -out=log2.txt
python classifStructuredL2.py -N=7 -m=20000 -hidden=[16,12,8] -opt=adam \
                           -kwargs={'learning_rate':1e-3} -out=log3.txt
"""
# Tasks at hand:
# Implement the classifier for an example data set for the structured l2 penalty
# Distinguish between results by comparing to lasso
import os
import time
import sys
import pickle
import argparse
import numpy as np
import tensorflow as tf
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .classifpredAnalysis import predclassif
from .processDataFiles import ProcessMyData
from .processIntergratedData import prep_data_2_train, main_74b, func_y
from .drawingNetsformatting import paramreshape
from .deepnetworkdiag import NeuralNetwork
from .py_lasso_l2 import func_classifier_l2l1, func_classifier_l1
from .all_loss_functions import func_cross_entropy_loss


# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# with fixed seed initial values for train-able variables and training data
# will be the same, so it is easier to compare optimization performance
SEED = 52
# you can try tf.float32/np.float32 data types
TF_DATA_TYPE = tf.float64
NP_DATA_TYPE = np.float64
# how frequently log is written and checkpoint saved
LOG_INTERVAL_IN_SEC = 0.05

# variants of activation functions
ACTIVATIONS = {'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid}

# variants of initializers
INITIALIZERS = {'xavier': tf.contrib.layers.xavier_initializer(seed=SEED),
                'rand_uniform': tf.random_uniform_initializer(seed=SEED),
                'rand_normal': tf.random_normal_initializer(seed=SEED)}

# variants of tensor flow built-in optimizers
TF_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}

# checkpoints are saved to <log_file_name>.ckpt
out_file = None
log_prev_time, log_first_time = None, None
# are used to continue log when script is started from a checkpoint
step_delta, time_delta = 0, 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', help='MLP hidden layers structure', type=str, default='[40, 10]')
    parser.add_argument('-a', '--activation', help='nonlinear activation function', type=str, choices=['relu', 'sigmoid', 'tanh'], default='tanh')
    parser.add_argument('-i', '--initializer', help='trainable variables initializer', type=str, choices=['rand_normal', 'rand_uniform', 'xavier'], default='xavier')
    parser.add_argument('-opt', '--optimizer', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'], default='lm')
    parser.add_argument('-kwargs', help='optimizer parameters', type=str, default="{'mu':5.,'mu_inc':10,'mu_dec':10,'max_inc':100}")
    parser.add_argument('-out', help='output log file name', type=str, default='log.txt')
    parser.add_argument("-c", "--cont", help="continue from checkpoint", action="store_true")
    args = parser.parse_args()
    hidden = eval(args.hidden)
    activation = ACTIVATIONS[args.activation]
    initializer = INITIALIZERS[args.initializer]
    optimizer = args.optimizer
    print(args.kwargs)
    kwargs = eval(args.kwargs)
    out = args.out
    use_checkpoint = args.cont
    return hidden, activation, initializer, optimizer, kwargs, out, use_checkpoint


# saves checkpoint and outputs current step/loss/mu to files
def log(step, loss, params, mu=None):
    global log_prev_time, log_first_time

    now = time.time()
    if log_prev_time and now - log_prev_time < LOG_INTERVAL_IN_SEC:
        return
    if not log_prev_time:
        log_prev_time, log_first_time = now, now
    secs_from_start = int(now - log_first_time) + time_delta
    step += step_delta
    message = step + secs_from_start + loss
    message += mu if mu else ''
    print(message)
    with open(out_file, 'a') as file:
        file.write('message \n')
    pickle.dump((step, secs_from_start, params), open(out_file + '.ckpt', "wb"))
    log_prev_time = now


# calculates Jacobian matrix for y with respect to x
def jacobian_classif(y, x):
    stopgrads = tf.where(x == 0)
    m = tf.shape(y)[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(TF_DATA_TYPE, size=m),
    ]
    _, jacobian_classif = tf.while_loop(lambda i, _: i < m, lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x, stop_gradients=stopgrads, unconnected_gradients='zero')[0])), loop_vars)
    dxdt = tf.gradients(tf.reduce_sum(tf.abs(x)), x, unconnected_gradients='zero')[0]
    print(jacobian_classif.stack())
    return jacobian_classif.stack(), dxdt
# performs network training and updates parameter values according to LM algorithm


def train_lm(feed_dict, loss, params, y_hat, lambda1, kwargspred, **kwargs): 
    r = loss
    mu1, _, mu_dec, max_inc = kwargs['mu'], kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    wb_shapes, wb_sizes_classif, hidden = kwargspred['wb_shapes'], kwargspred['wb_sizes'], kwargspred['hidden']
    activation, xydat= kwargspred['activation'], kwargspred['xydat']
    hess_approx=True
    print(wb_shapes)
    print(wb_sizes_classif)
    neurons_cnt = params.shape[0].value
    mu_current = tf.placeholder(TF_DATA_TYPE, shape=[1])
    imatrx = tf.eye(neurons_cnt, dtype=TF_DATA_TYPE)
    y_hat_flat = tf.squeeze(y_hat)

    if hess_approx:
        jcb = jacobian(y_hat_model, p)
        print(tf.transpose(jcb).shape)
        j1 = jacobian_mse(y_hat_model, p, nm_set_points, all_sizes_vec, all_shapes_vec)
        jt = tf.transpose(j1)
        l2_grad = tf.gradients(l2_norm_val, l2_p)[0]
        dxdt = tf.expand_dims(tf.gradients(all_reg0, lassop)[0], 1)
        hess_l2_ps = tf.hessians(l2_norm_val, l2_p)[0]
        print('The shape is;', j1.shape)
        jtj1 = tf.matmul(jt, j1)
        jtr1 = 2*tf.matmul(jt, r1)
        l2grad = tf.expand_dims(l2_grad, 1)
        s_l2grad = tf.matmul(l2grad, tf.transpose(l2grad))
        # compute gradient of l2 params
        reshaped_gradl2 = jtr1[0:shaped_new]
        reshaped_l20 = reshaped_gradl2 + lambda_param2 * l2grad # l2_p_grads, 1)
        # build another hessian
        jt_hess = jt[0:shaped_new, :] + lambda_param2 * l2grad # l2_p_grads, 1)
        jt_hess_end = tf.concat([jt_hess, jt[shaped_new:, :]], axis=0)
        j1_t = tf.transpose(jt_hess_end)
        # calculate gradient for lasso params group
        reshaped_gradl1 = jtr1[shaped_new:]
        reshaped_gradl0 = reshaped_gradl1 + lambda_param * dxdt # tf.expand_dims(dxdt, 1) #tf.sign(lasso_p), 1)
        # Assemble the lasso group
        jtj = tf.matmul(jt_hess_end, j1_t)
        jtr = tf.concat([reshaped_l20, reshaped_gradl0], axis=0)
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))
        # The other hess using hessian for in --> hid1
        hess_part2 = jtj1[0:shaped_new, 0:shaped_new] + s_l2grad #hess_l2_ps# + h_mat_l2
        hess_partsconc = tf.concat([hess_part2, jtj1[0:shaped_new, shaped_new:]], axis=1)
        jtj3 = tf.concat([hess_partsconc, jtj1[shaped_new:, :]], axis=0)
        # remove it
        # j, dxdt = jacobian_classif(y_hat_flat, params)
        # j_t = tf.transpose(j)
        # hess = tf.matmul(j_t, j)
        # g0 = tf.matmul(j_t, r)
        # print('Shape is: ')
        # print(j)
        # g = g0 # + lambda1  * tf.reshape(dxdt, shape=(neurons_cnt, 1))
    else:
        hess = tf.hessians(loss, params)[0]
        g = -tf.gradients(loss, params)[0]
        g = tf.reshape(g, shape=(neurons_cnt, 1))

    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=TF_DATA_TYPE))
    hess_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=TF_DATA_TYPE))
    g_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=TF_DATA_TYPE))
    save_params = tf.assign(p_store, params)
    restore_params = tf.assign(params, p_store)
    save_hess_g = [tf.assign(hess_store, hess), tf.assign(g_store, g)]
    input_mat = hess_store + tf.multiply(mu_current, imatrx)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat), g_store)
    except:
        c = tf.constant(0.1)
        input_mat += np.identity((input_mat.shape)) * c
        dx = tf.matmul(tf.linalg.inv(input_mat), g_store)
    dx = tf.squeeze(dx)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1)
    lm = opt.apply_gradients([(-dx, params)])

    feed_dict[mu_current] = np.array([mu1])
    # session is an object of  ten flow Session
    with tf.Session() as session:
        step = 0
        matpvals = []
        session.run(tf.global_variables_initializer())
        current_loss = session.run(loss, feed_dict)
        while current_loss > 1e-10 and step < 400:
            step += 1
            log(step, current_loss, session.run(params), feed_dict[mu_current][0])
            session.run(save_params)
            session.run(save_hess_g, feed_dict)
            # session.run(hess_store, feed_dict)
            success = False
            for _ in range(max_inc):
                session.run(save_hess_g, feed_dict)
                session.run(hess_store, feed_dict)
                session.run(lm, feed_dict)
                p0 = tf.where(tf.math.equal(params, 0), tf.zeros_like(params), params)
                new_loss = session.run(loss, feed_dict)
                if new_loss < current_loss:
                    # Break the params into two groups: save those with lasso as before
                    # And use the other group for structured l2
                    numhidbiasparams = wb_sizes_classif[0] + wb_sizes_classif[1]
                    lassop0 = p0[numhidbiasparams:]
                    matpvals.append(lassop0)
                    in2hidparams = p0[0:numhidbiasparams]
                    if len(matpvals) == 3:
                        print(step)
                        sgn1 = tf.multiply(matpvals[0], matpvals[1])
                        sgn2 = tf.multiply(matpvals[1], matpvals[2])
                        last_tensor_vec = matpvals[2]
                        px = tf.where(tf.math.logical_and(sgn2 < 0, sgn1 < 0), tf.zeros_like(last_tensor_vec), last_tensor_vec)
                        sgn01 = session.run(sgn1)
                        sgn02 = session.run(sgn2)
                        oscvec = np.where((sgn01 < 0) & (sgn02 < 0))
                        # in2hidparams = params_grouping_func(ind2hidparams, wb_sizes_classif, wb_shapes_classif)
                        print(in2hidparams, px)
                        px0 = tf.concat([in2hidparams, px], 0)
                        params.assign(px0)
                        matpvals=[]
                    else:
                        params.assign(p0)
                    session.run(save_params)
                    session.run(restore_params)
                    feed_dict[mu_current] /= mu_dec
                    current_loss = new_loss
                    success = True
                    break
                else:
                    feed_dict[mu_current] *= mu_dec
                    session.run(restore_params)
            if not success:
                print('Failed to improve')
                break
        correct_prediction, feed_dict2 = predclassif(wb_sizes_classif, xydat, hidden, params, activation, wb_shapes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # accuracy metric using session of run of accuracy
        pnews =  session.run(restore_params)
        pnews[pnews < 1e-06] = 0
        pnonzero = np.count_nonzero(pnews)
        # print(wb_shapes, wb_sizes_classif)
        # paramreshape(pnonzero, wb_shapes, wb_sizes_classif)
        print('Non-zeros:')
        print(pnonzero)
        print('ENDED ON STEP:, FINAL LOSS:')
        print(step, current_loss)
        print("Accuracy:", session.run(accuracy, feed_dict2))
    return pnews


def train_tf_classifier(feed_dict1, params, loss, train_step, logits, labels_one_hot, feed_dict2):
    step = 0
    wbsize, wbshape = 1, 1
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        current_loss = session.run(loss, feed_dict1)
        while current_loss > 1e-10 and step < 400:
            step += 1
            log(step, current_loss, session.run(params))
            session.run(train_step, feed_dict1)
            current_loss = session.run(loss, feed_dict1)
            print("Epoch: {0} ; training loss: {1}".format(step, loss))
        print('training finished')
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(feed_dict2))
        return correct_prediction, wbshape, wbsize


def build_mlp_structure_classify(n, nclasses, mlp_hidden_structure):
    mlp_structure = [n] + mlp_hidden_structure+[nclasses]
    wb_shapes_classif = []
    for idx in range(len(mlp_hidden_structure) + 1):
        wb_shapes_classif.append((mlp_structure[idx], mlp_structure[idx+1]))
        wb_shapes_classif.append((1, mlp_structure[idx+1]))
    wb_sizes_classif = [hclassif * wclassif for hclassif, wclassif in wb_shapes_classif]
    neurons_cnt_classif = sum(wb_sizes_classif)
    print('Total number of trainable parameters is', neurons_cnt_classif)
    return neurons_cnt_classif, wb_shapes_classif, wb_sizes_classif


def main_classif(xtrain0, ytr, nclasses, xtest, ytest, sess, choose_flag):
    global out_file, step_delta, time_delta
    hidden, activation, initializer, optimizer, kwargs, out_file, use_checkpoint = parse_arguments()
    xtrain00 = np.asanyarray(xtrain0)
    biases = np.ones((xtrain00.shape[0], 1))
    # xtr = np.c_[xtrain00, biases]
    xtr = xtrain00

    xtest00 = np.asanyarray(xtest)
    biases1 = np.ones((xtest00.shape[0], 1))
    # xtest1 = np.c_[xtest00, biases1]
    xtest1 = xtest00
    n = xtr.shape[1]
    m = xtr.shape[0]
    neurons_cnt, wb_shapes, wb_sizes = build_mlp_structure_classify(n, nclasses, hidden)

    ckpt_data = None
    if use_checkpoint and os.path.exists(out_file + '.ckpt'):
        step_delta, time_delta, ckpt_data = pickle.load(open(out_file + '.ckpt', "rb"))
    else:
        with open(out_file, "a") as file:
            file.write('" ".join(sys.argv[1:])} \n')
    kwargs1 = {'n': n, 'm': m, 'hidden': hidden, 'activation': activation, 'nclasses': nclasses,
               'initializer': initializer, 'neurons_cnt':neurons_cnt, 'wb_shapes': wb_shapes}

    # loss, params, x, y, y_hat, l2_normed = build_tf_nn(wb_sizes, ckpt_data, **kwargs1)
    # feed dictionary of dp_x and dp_y
    # loss, params, x, y, y_hat, l2_normed = func_cross_entropy_loss(wb_sizes, ckpt_data, **kwargs1)
    # feed_dict = {x:xtr, y:ytr}
    # feed_dict2 = {x: xtest1, y: ytest}
    xydat = [xtest1, ytest]
    xydatrain = [xtr, ytr]
    opt_obj = tf.train.GradientDescentOptimizer(learning_rate=1)
    kwargspred = {'wb_shapes': wb_shapes, 'wb_sizes': wb_sizes, 'hidden': hidden,
    'activation': activation, 'xydat': xydat, 'xtr': xtr, 'xydatrain': xydatrain,
    'ytr': ytr, 'sess': sess, 'neurons_cnt': neurons_cnt, 'opt_obj': opt_obj, 'choose_flag': choose_flag}

    if optimizer == 'lm':
        # restoreparams = train_lm(feed_dict, loss, params, y_hat, l2_normed, kwargspred, **kwargs)
        restoreparams, correct_pred, ypredtrained = func_classifier_l2l1(xtest1, ytest, kwargs1, kwargspred, **kwargs)
        # restoreparams, correct_pred = func_classifier_l1(xtest1, ytest, kwargs1, kwargspred, **kwargs)
        # predclassif(wb_sizes, xydat, hidden, restoreparams, activation, wb_shapes, nclasses)
    else:
        train_step = TF_OPTIMIZERS[optimizer](0.1).minimize(loss)
        train_tf_classifier(feed_dict, params, loss, train_step, y, y_hat,feed_dict2)
    return restoreparams, wb_shapes, wb_sizes, hidden, correct_pred, ypredtrained
# service@eeteuropart.de


def build_tf_nn(wb_sizes_classif, ckpt_data, **kwargs1):
    # placeholder variables (we have m data points)
    nclassif, hidden, activation = kwargs1['n'], kwargs1['hidden'], kwargs1['activation'],
    initializer, neurons_cnt_classif, wb_shapes = kwargs1['initializer'], kwargs1['neurons_cnt'], kwargs1['wb_shapes']
    # n labels is for 2 # number of output classes or labels
    xclassif = tf.placeholder(tf.float64, shape=[None, nclassif])
    labels = tf.placeholder(tf.int64, shape = [None, ])
    yclassif = tf.one_hot(labels, 2)
    if ckpt_data is not None:
        params = tf.Variable(ckpt_data, dtype=TF_DATA_TYPE)
    else:
        params = tf.Variable(initializer([neurons_cnt_classif], dtype=tf.float64))
    classif_tensors = tf.split(params, wb_sizes_classif, 0)
    for i in range(len(classif_tensors)):
        classif_tensors[i] = tf.reshape(classif_tensors[i], wb_shapes[i])
    ws_classif = classif_tensors[0:][::2]
    bs_classif = classif_tensors[1:][::2]
    y_hat_classif = xclassif
    for i in range(len(hidden)):
        y_hat_classif = activation(tf.matmul(y_hat_classif, ws_classif[i]) + bs_classif[i])
    y_hat_classif = tf.matmul(y_hat_classif, ws_classif[-1]) + bs_classif[-1]
    ###################################################################################
    lambda1 = 0.0005
    regparam = lambda1 * tf.reduce_sum(tf.abs(params))
    ###################################################################################
    lambda2 = 0.00
    nhidbiasparams = wb_sizes_classif[0] + wb_sizes_classif[1]
    in2hiddenparams = params[0:wb_sizes_classif[0]]
    b1matrix = params[wb_sizes_classif[0]:nhidbiasparams]
    # structuredl2pen = structuredl2norm(in2hiddenparams, b1matrix)
    structuredl2pen = 0
    # regparam2 = lambda2 * structuredl2pen
    ####################################################################################
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat_classif, labels=yclassif)) + regparam # + regparam2
    return loss, params, xclassif, yclassif, y_hat_classif, lambda1
