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
import os
import sys
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .processDataFiles import ProcessMyData
from .classifpredAnalysis import func_pred
from .deepnetworkdiag import NeuralNetwork
from .drawingNetsformatting import paramreshape
from .processIntergratedData import prep_data_2_train, mainfile

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# with fixed seed initial values for train-able variables and training data
# will be the same, so it is easier to compare optimization performance
SEED = 10
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

# sudo apt install wicd-gtk - installed!
# sudo apt remove network-manager-gnome network-manager
# sudo dpkg --purge network-manager-gnome network-manager


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', help='MLP hidden layers structure',
    type=str,default='[10,40]')
    parser.add_argument('-a', '--activation', help='nonlinear activation function', type=str, choices=['relu', 'sigmoid', 'tanh'], default='tanh')
    parser.add_argument('-i', '--initializer', help='trainable variables initializer', type=str, choices=['rand_normal', 'rand_uniform', 'xavier'], default='xavier')
    parser.add_argument('-opt', '--optimizer', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'], default='lm')
    parser.add_argument('-kwargs', help='optimizer parameters', type=str, default="{'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':100}")
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
def log(step, loss, parms, mu=None):
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
        file.write(message + '\n')
    pickle.dump((step, secs_from_start, parms),
                open(out_file + '.ckpt', "wb"))
    log_prev_time = now

# calculates Jacobian matrix for y with respect to x

def autogrdgradient(wparmslist):
    pass


def jacobian(y, x):
    stopgrads = tf.where(x == 0)
    m = y.shape[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(TF_DATA_TYPE, size=m),
    ]
    _, j = tf.while_loop(lambda i, _: i < m, lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x, stop_gradients=stopgrads, unconnected_gradients='zero')[0])), loop_vars)
    dxdt = tf.gradients(tf.reduce_sum(tf.abs(x)), x, unconnected_gradients='zero')[0]
    print(j.stack())
    return j.stack(), dxdt

# performs network training and updates parameter values according to LM algorithm


def train_lm(feed_dict, loss, r, parms, y_hat, wb_sizes, **kwargs):
    mu, lambda1 = kwargs['mu'], kwargs['lambda1']
    mu_inc, mu_dec, max_inc = kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    hess_approx = True
    neurons_cnt = parms.shape[0].value
    mu_current = tf.placeholder(TF_DATA_TYPE, shape=[1])
    imatrx = tf.eye(neurons_cnt, dtype=TF_DATA_TYPE)
    y_hat_flat = tf.squeeze(y_hat)

    if hess_approx:
        j, dxdt = jacobian(y_hat_flat, parms)
        j_t = tf.transpose(j)
        hess = tf.matmul(j_t, j)
        g = tf.matmul(j_t, r)
        # g = gt + lambda1 * tf.reshape(dxdt, shape=(neurons_cnt, 1))
        
    else:
        stopgrads = tf.where(tf.math.equal(parms, 0))
        hess = tf.hessians(loss, parms)[0]
        g = -tf.gradients(loss, parms)[0]
        # g = -tf.gradients(loss, parms, stop_gradients=stopgrads, unconnected_gradients = 'zero')[0]
        g = tf.reshape(g, shape=(neurons_cnt, 1))

    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=TF_DATA_TYPE))
    hess_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=TF_DATA_TYPE))
    g_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=TF_DATA_TYPE))
    save_parms = tf.assign(p_store, parms)
    restore_parms = tf.assign(parms, p_store)
    save_hess_g = [tf.assign(hess_store, hess), tf.assign(g_store, g)]
    input_mat = hess_store + tf.multiply(mu_current, imatrx)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat), g_store)
    except tf.linalg.LinAlgError:
        c = tf.constant(0.1)
        input_mat += np.identity((input_mat.shape)) * c
        dx = tf.matmul(tf.linalg.inv(input_mat), g_store)
    dx = tf.squeeze(dx)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    lm = opt.apply_gradients([(-dx, parms)])

    feed_dict[mu_current] = np.array([mu])
    session = tf.Session()
    step = 0
    matpvals = []
    session.run(tf.global_variables_initializer())
    current_loss = session.run(loss, feed_dict)
    while current_loss > 1e-10 and step < 200:
        step += 1
        log(step, current_loss, session.run(parms), feed_dict[mu_current][0])
        session.run(save_parms)
        session.run(save_hess_g, feed_dict)
        # session.run(hess_store, feed_dict)
        success = False
        for _ in range(max_inc):
            session.run(save_hess_g, feed_dict)
            session.run(hess_store, feed_dict)
            session.run(lm, feed_dict)
            p0 = tf.where(tf.math.equal(parms, 0), tf.zeros_like(parms), parms)
            new_loss = session.run(loss, feed_dict)
            if new_loss < current_loss:
                # Determine the groups to penalize between in->hidden and lasso
                numhidbiasparms = wb_sizes[0] + wb_sizes[1]
                lassop0 = p0[numhidbiasparms:]
                # matpvals.append(lassop0)
                in2hidparms = p0[0:numhidbiasparms]
                matpvals.append(p0)
                if len(matpvals) == 3:
                    print(step)
                    sgn1 = tf.multiply(matpvals[0], matpvals[1])
                    sgn2 = tf.multiply(matpvals[1], matpvals[2])
                    last_tensor_vec = matpvals[2]
                    px = tf.where(tf.math.logical_and(sgn2 < 0, sgn1 < 0),
                    tf.zeros_like(last_tensor_vec), last_tensor_vec)
                    oscvec = tf.where(tf.math.logical_and(sgn1 < 0, sgn2 < 0))
                    print(session.run(oscvec))
                    px0 = tf.concat([in2hidparms, px], 0)
                    # parms.assign(px0)
                    parms.assign(px)
                    matpvals = []
                    # matpvals.append(px)
                else:
                    parms.assign(p0)
                session.run(save_parms)
                session.run(restore_parms)
                feed_dict[mu_current] /= mu_dec
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu_current] *= mu_inc
                session.run(restore_parms)
        if not success:
            print('Failed to improve')
            break
    pnew = session.run(restore_parms)
    pnew[pnew < 1e-05] = 0
    pnonzero = np.count_nonzero(pnew)
    print(f'ENDED ON STEP: {step}, FINAL LOSS: {current_loss}')
    print(f'Number of nonzero parameters: {pnonzero}')
    return pnew


def train_tf(feed_dict, parms, loss, train_step):
    step = 0
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calculate initial loss
    current_loss = session.run(loss, feed_dict)
    while current_loss > 1e-10 and step < 400:
        step += 1
        log(step, current_loss, session.run(parms))
        session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)


def build_mlp_structure(n, mlp_hidden_structure):
    mlp_structure = [n] + mlp_hidden_structure + [1]
    wb_shapes = []
    for i in range(len(mlp_hidden_structure) + 1):
        wb_shapes.append((mlp_structure[i], mlp_structure[i + 1]))
        wb_shapes.append((1, mlp_structure[i + 1]))
    wb_sizes = [h * w for h, w in wb_shapes]
    neurons_cnt = sum(wb_sizes)
    print(f'Total number of trainable parameters is {neurons_cnt}')
    return neurons_cnt, wb_shapes, wb_sizes


def main(xtrain0, ytr0):
    global out_file, step_delta, time_delta
    hidden, activation, initializer, optimizer, kwargs, out_file, use_checkpoint=parse_arguments()

    xtrain00 = np.asanyarray(xtrain0)
    print(xtrain00.shape[0])
    biases = np.ones((xtrain00.shape[0], 1))
    xtr = np.c_[xtrain00, biases]

    ytr = ytr0[:, None]

    n = xtr.shape[1]
    m = xtr.shape[0]

    neurons_cnt, wb_shapes, wb_sizes = build_mlp_structure(n, hidden)

    ckpt_data = None
    if use_checkpoint and os.path.exists(out_file + '.ckpt'):
        step_delta, time_delta, ckpt_data = pickle.load(open(out_file + '.ckpt', "rb"))
    else:
        with open(out_file, "a") as file:
            file.write(f'{" ".join(sys.argv[1:])}\n')

    kwargs1 = {'n': n, 'm': m, 'hidden': hidden, 'activation': activation,
               'initializer': initializer, 'neurons_cnt': neurons_cnt, 'wb_shapes': wb_shapes}

    loss, parms, r, x, y, y_hat, lambda1 = build_tf_nn(wb_sizes, ckpt_data, **kwargs1)

    kwargs['lambda1'] = lambda1
    # feed dictionary of dp_x and dp_y
    feed_dict = {x: xtr, y: ytr}
    if optimizer == 'lm':
        restoreparms = train_lm(feed_dict, loss, r, parms, y_hat, wb_sizes, **kwargs)
    else:
        train_step = TF_OPTIMIZERS[optimizer](**kwargs).minimize(loss)
        train_tf(feed_dict, parms, loss, train_step)
    return restoreparms, wb_shapes, wb_sizes, activation, hidden


def build_tf_nn(wb_sizes, ckpt_data, **kwargs1):
    # placeholder variables (we have m data points)
    n, m, hidden, activation = kwargs1['n'], kwargs1['m'], kwargs1['hidden'], kwargs1['activation'],
    initializer, neurons_cnt, wb_shapes = kwargs1['initializer'], kwargs1['neurons_cnt'], kwargs1['wb_shapes']
    x = tf.placeholder(TF_DATA_TYPE, shape=[m, n])
    y = tf.placeholder(TF_DATA_TYPE, shape=[m, 1])
    if ckpt_data is not None:
        parms = tf.Variable(ckpt_data, dtype=TF_DATA_TYPE)
    else:
        parms = tf.Variable(initializer([neurons_cnt], dtype=TF_DATA_TYPE))
    tensors = tf.split(parms, wb_sizes, 0)
    for i in range(len(tensors)):
        tensors[i] = tf.reshape(tensors[i], wb_shapes[i])
    ws = tensors[0:][::2]
    bs = tensors[1:][::2]
    y_hat = x
    for i in range(len(hidden)):
        y_hat = activation(tf.matmul(y_hat, ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, ws[-1]) + bs[-1]
    r = y - y_hat
    lambda1 = 0.00
    regparam = lambda1 * tf.reduce_sum(tf.abs(parms))
    loss = tf.reduce_mean(tf.square(r)) + regparam
    return loss, parms, r, x, y, y_hat, lambda1


if __name__ == '__main__':
    mlabfile = '/home/nath/forLenovoUbuntu/datfile/sparsedat/testData.mat'
    xtrain0, xt, ytr0, yt = ProcessMyData().load_solar_data(mlabfile, 'testData')
    # ################## cancer data #####################################
    a_file = '/home/nath/LM_code_final/data/heartdisease/Integrated.csv'
    smallfile_feats = '/home/nath/LM_code_final/data/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    dat74features, _ = mainfile(a_file, smallfile_feats)
    formatted_data, x00_train, y_train, test_x, test_y = prep_data_2_train(dat74features)
    xvalues = x00_train.values
    yvalues = y_train.values
    # yvalues = ytr[:, None]

    parms0, wbshapes, wbsizes, activation, nhidden = main(xvalues, yvalues)
    parms = np.asarray(parms0)
    # ################## on the test set #############################################
    predy = func_pred(nhidden, parms0, wbsizes, wbshapes, activation, xvalues)
    # predy = func_pred(nhidden, parms0, wbsizes, wbshapes, activation, xtrain0)
    f = plt.figure(1)
    nmtest = xvalues.shape[0]
    colors = np.random.rand(nmtest)
    area = (10 * np.random.rand(nmtest))**2  # 0 to 15 point radius
    plt.scatter(yvalues, predy, s=area, c=colors, alpha=0.5)
    allnetworkweights = paramreshape(parms, wbshapes, wbsizes, nhidden)
    network = NeuralNetwork()
    # ########################### Attach layers and number of neurons # and weights ######
    for idxparms in allnetworkweights:
        network.add_layer(idxparms.shape[1], idxparms)
        print(idxparms.shape)
    # ########### last hidden to output ################################################
    nh = np.ones((1, allnetworkweights[-1].shape[0]))
    network.add_layer(nh.shape[1], nh)
    print(allnetworkweights[0])
    g = plt.figure(2)
    network.draw()
