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
__path__=[os.path.dirname(os.path.abspath(__file__))]
from .classifpredAnalysis import predclassif
from .processDataFiles import ProcessMyData


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', help='Rosenbrock function dimensionality', type=int, default=4)
    parser.add_argument('-m', help='number of random data points', type=int, default=5000)
    parser.add_argument('-hidden', help='MLP hidden layers structure', type=str, default='[20]')
    parser.add_argument('-a', '--activation', help='nonlinear activation function', type=str, choices=['relu', 'sigmoid', 'tanh'], default='tanh')
    parser.add_argument('-i', '--initializer', help='trainable variables initializer', type=str, choices=['rand_normal', 'rand_uniform', 'xavier'], default='xavier')
    parser.add_argument('-opt', '--optimizer', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'], default='lm')
    parser.add_argument('-kwargs', help='optimizer parameters', type=str, default="{'mu':5.,'mu_inc':10,'mu_dec':10,'max_inc':100}")
    parser.add_argument('-out', help='output log file name', type=str, default='log.txt')
    parser.add_argument("-c", "--cont", help="continue from checkpoint", action="store_true")
    args = parser.parse_args()
    n = args.N
    m = args.m
    hidden = eval(args.hidden)
    activation = ACTIVATIONS[args.activation]
    initializer = INITIALIZERS[args.initializer]
    optimizer = args.optimizer
    print(args.kwargs)
    kwargs = eval(args.kwargs)
    out = args.out
    use_checkpoint = args.cont
    return n, m, hidden, activation, initializer, optimizer, kwargs, out, use_checkpoint

# prepare the network number of weights, and shapes
# using hidden layer node as param

def build_classify_network_model(n):
    mm = [10]
    st_classif = [n] + mm + [2]
    shapes_classif = []
    for idx in range(len(mm) + 1):
        shapes_classif.append((st_classif[idx], st_classif[idx + 1]))
        shapes_classif.append((1, st_classif[idx + 1]))
    sizes_classif = [hclassif * wclassif for hclassif, wclassif in shapes_classif]
    neurons_cnt_classif = sum(sizes_classif)
    return shapes_classif, sizes_classif, neurons_cnt_classif, mm

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
    message = f'{step} {secs_from_start} {loss}'
    message += f' {mu}' if mu else ''
    print(message)
    with open(out_file, 'a') as file:
        file.write(message + '\n')
    pickle.dump((step, secs_from_start, params),
                open(out_file + '.ckpt', "wb"))
    log_prev_time = now

# calculates Jacobian matrix for y with respect to x
def jacobian(y, x):
    m = y.shape[0]
    loop_vars = [
        tf.constant(0, tf.int32), tf.TensorArray(TF_DATA_TYPE, size=m),]
    _, j = tf.while_loop(
        lambda i, _: i < m,
        lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x)[0])), loop_vars)
    dxdt = tf.gradients(tf.reduce_sum(tf.abs(x)), x)[0]
    return j.stack(), dxdt


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


# performs network training and updates parameter values according to LM algorithm

def train_lm(feed_dict, loss, params, y_hat, lambda1, kwargspred, **kwargs): 
    r = loss
    mu1, _, mu_dec, max_inc = kwargs['mu'], kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    wb_shapes, wb_sizes_classif, hidden = kwargspred['wb_shapes'], kwargspred['wb_sizes'], kwargspred['hidden']
    activation, xydat= kwargspred['activation'], kwargspred['xydat']
    hess_approx=False
    neurons_cnt = params.shape[0].value
    mu_current = tf.placeholder(TF_DATA_TYPE, shape=[1])
    imatrx = tf.eye(neurons_cnt, dtype=TF_DATA_TYPE)
    y_hat_flat = tf.squeeze(y_hat)

    if hess_approx:
        j, dxdt = jacobian(y_hat_flat, params)
        j_t = tf.transpose(j)
        hess = tf.matmul(j_t, j)
        g0 = tf.matmul(j_t, r)
        print('Shape is: ')
        print(params.get_shape())
        g = g0 + lambda1  * tf.reshape(dxdt, shape=(neurons_cnt, 1))
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
            #session.run(hess_store, feed_dict)
            success = False
            for _ in range(max_inc):
                session.run(save_hess_g, feed_dict)
                session.run(hess_store, feed_dict)
                session.run(lm, feed_dict)
                p0 = tf.where(tf.math.equal(params, 0), tf.zeros_like(params), params)
                new_loss = session.run(loss, feed_dict)
                if new_loss < current_loss:
                    matpvals.append(p0)
                    if len(matpvals) == 3:
                        print(step)
                        sgn1 = tf.multiply(matpvals[0], matpvals[1])
                        sgn2 = tf.multiply(matpvals[1], matpvals[2])
                        last_tensor_vec = matpvals[2]
                        px = tf.where(tf.math.logical_and(sgn2 < 0, sgn1 < 0), tf.zeros_like(last_tensor_vec), last_tensor_vec)
                        sgn01 = session.run(sgn1)
                        sgn02 = session.run(sgn2)
                        oscvec = np.where((sgn01 < 0) & (sgn02 < 0))
                        print(oscvec)
                        params.assign(px)
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
        # acc=session.run(accuracy)
        print(f'ENDED ON STEP: {step}, FINAL LOSS: {current_loss}')
        print(f'Parameters: {session.run(restore_params)}')
        # print("Accuracy:", acc.eval(feed_dict2))
        print("Accuracy:", session.run(accuracy, feed_dict2))
    return restore_params



def train_tf_classifier(feed_dict1, params, loss, train_step, logits, labels_one_hot, feed_dict2):
    step = 0
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        current_loss = session.run(loss, feed_dict1)
        while current_loss > 1e-10 and step < 400:
            step += 1
            log(step, current_loss, session.run(params))
            session.run(train_step, feed_dict1)
            current_loss = session.run(loss, feed_dict1)
            # print("Epoch: {0} ; training loss: {1}".format(step, loss))
        print('training finished')
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(feed_dict2))

def build_mlp_structure_classify(n, nclasses, mlp_hidden_structure):
    mlp_structure = [n] + mlp_hidden_structure+[nclasses]
    wb_shapes_classif = []
    for idx in range(len(mlp_hidden_structure) + 1):
        wb_shapes_classif.append((mlp_structure[idx], mlp_structure[idx+1]))
        wb_shapes_classif.append((1, mlp_structure[idx+1]))
    wb_sizes_classif = [hclassif * wclassif for hclassif, wclassif in wb_shapes_classif]
    neurons_cnt_classif = sum(wb_sizes_classif)
    print(f'Total number of trainable parameters is {neurons_cnt_classif}')
    return neurons_cnt_classif, wb_shapes_classif, wb_sizes_classif

def main(xtrain0, ytr, nclasses, xtest, ytest):
    global out_file, step_delta, time_delta
    _, _, hidden, activation, initializer, optimizer, kwargs, out_file, use_checkpoint = parse_arguments()
    
    xtrain00 = np.asanyarray(xtrain0)
    biases = np.ones((xtrain00.shape[0], 1))
    xtr = np.c_[xtrain00, biases]

    xtest00 = np.asanyarray(xtest)
    biases = np.ones((xtest00.shape[0], 1))
    xtest1 = np.c_[xtest00, biases]
    
    print(ytr.shape)
    n = xtr.shape[1]
    m = xtr.shape[0]
    
    
    neurons_cnt, wb_shapes, wb_sizes = build_mlp_structure_classify(n, nclasses, hidden)

    ckpt_data = None
    if use_checkpoint and os.path.exists(out_file + '.ckpt'):
        step_delta, time_delta, ckpt_data = pickle.load(open(out_file + '.ckpt', "rb"))
    else:
        with open(out_file, "a") as file:
            file.write(f'{" ".join(sys.argv[1:])}\n')
    
    kwargs1 = {'n':n, 'm':m, 'hidden':hidden, 'activation': activation, 
               'initializer': initializer, 'neurons_cnt':neurons_cnt, 
               'wb_shapes':wb_shapes}

    loss, params, x, y, y_hat, lambda1 = build_tf_nn(wb_sizes, ckpt_data, **kwargs1)
    # feed dictionary of dp_x and dp_y
    feed_dict = {x:xtr, y:ytr}
    feed_dict2 = {x: xtest1, y: ytest}
    xydat = [xtest1, ytest]

    kwargspred = {'wb_shapes':wb_shapes, 'wb_sizes':wb_sizes, 'hidden':hidden, 
    'activation': activation, 'feed_dict2': feed_dict2, 'xydat':xydat}

    if optimizer == 'lm':
        restoreparams = train_lm(feed_dict, loss, params, y_hat, lambda1, kwargspred, **kwargs)
        predclassif(wb_sizes, xydat, hidden, restoreparams, activation, wb_shapes)
    else:
        train_step = TF_OPTIMIZERS[optimizer](0.1).minimize(loss)
        train_tf_classifier(feed_dict, params, loss, train_step, y, y_hat,feed_dict2)


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
    lambda1 = 0.00
    regparam = lambda1 * tf.reduce_sum(tf.abs(params))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat_classif, labels=yclassif)) + regparam
    return loss, params, xclassif, yclassif, y_hat_classif, lambda1
    


if __name__ == '__main__':
    dfiletowrite = '~/UCIdata/'
    heartdata = '~/tfCodes/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    #datarray, yarray = ProcessMyData().heartdiseasedata(heartdata)
    #xy_train, xy_test, xy_valid = ProcessMyData().xtraintestdata(datarray, yarray, dfiletowrite)
    #nclasses = 1
    #x_train1, y_train1 = ProcessMyData().gettheclassifdata(heartdata)
    #classes = ytrain1.shape[1]
    # main(x_train1, y_train1, nclasses2)
    # # kwargs_dict_2 = {'neurons_cnt_classify': neurons_cnt_classify, 'sizes_classify': sizes_classify,
    #               'shapes_classify': shapes_classify}
    # shapes_classify, sizes_classify, neurons_cnt_classify, mm = build_classify_network_model(n)
    xtrainingdat, ylabeldat = ProcessMyData().gettheclassifdata(heartdata)
    nclasses3 = ylabeldat.shape[1]
    print(ylabeldat.shape)
    x1tr, y1tr = ProcessMyData().cancerdatadiagnosislabel(heartdata)
    trainx, trainy, testx, testy = ProcessMyData().processcancerdata(x1tr, y1tr)
    main(trainx, trainy, nclasses3, testx, testy)
    # main(xtrainingdat, ylabeldat, nclasses3)
