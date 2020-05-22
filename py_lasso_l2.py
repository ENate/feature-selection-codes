import os
import math
import csv
import numpy as np
import tensorflow as tf
from itertools import combinations, count
from tensorflow.python.ops.parallel_for.gradients import jacobian
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from .func_cond import func_compute_cond
from .classifpredAnalysis import predclassif, func_pred_new
from .py_func_loss import func_collect_allparams
from .all_loss_functions import func_cross_entropy_loss
from .all_loss_functions import func_mse_loss, func_mse_l2, func_structured_l2pen


def new_structured_l2(ps, allsizes, allshapes):
    shaped_new = np.int(allsizes[0]) + np.int(allsizes[1])
    idx1 = np.int(allshapes[0][0]) + np.int(allshapes[1][0])
    lasso_p = ps[shaped_new:]
    l2_p = ps[0:shaped_new]
    # ws0 = tf.reshape(ps[0:allsizes[0]], shape=allshapes[0])
    # bs0 = tf.reshape(ps[allsizes[0]:], allshapes[1])
    # concat_wb = tf.concat([ws0, bs0], axis=0)
    concat_wb = tf.reshape(l2_p, shape=(idx1, allshapes[0][1]))
    print(concat_wb)
    p_square = tf.square(concat_wb)
    xsum_cols = tf.reduce_sum(p_square, axis=1)
    xsum_sqrt = tf.sqrt(xsum_cols)
    print(xsum_cols)
    xsum_row = tf.reduce_sum(xsum_sqrt, axis=0)
    print(xsum_row)
    all_reg_0 = tf.reduce_sum(tf.abs(lasso_p))
    print(xsum_row)
    # hess_l2 = hessian_multivar(xsum_row, [ps])
    return xsum_row, all_reg_0


def jacobian_mse(y, x_var, n_m, all_sizesvec, all_shapesvec):
    """compute the Jacbian matrix """
    loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(tf.float64, size=n_m), ]
    _, jacobian = tf.while_loop(lambda i, _: i < n_m,
        lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x_var)[0])), loop_vars)
    # print(jacobian.shape)
    shaped_new0 = np.int(all_sizesvec[0]) + np.int(all_sizesvec[1])
    lasso_p = x_var[shaped_new0:]
    l2_ps = x_var[0:shaped_new0]
    l2_norm_val, _ = new_structured_l2(x_var, all_sizesvec, all_shapesvec)

    dxdt0 = tf.gradients(tf.reduce_sum(tf.abs(lasso_p)), lasso_p)[0]
    print(l2_norm_val)
    # hess_l2_ps = tf.hessians(l2_norm_val, l2_ps)[0]
    # hess_l2_ps2 = hessian_multivar(l2_norm_val, l2_ps)
    return jacobian.stack()#, dxdt0, hess_l2_ps, hess_l2_ps2, grad_l2


def jacobianhess(y, x, tf_loop=False):
    # If the shape of Y is fully defined you can choose between a
    # Python-level or TF-level loop to make the Jacobian matrix
    # If the shape of Y is not fully defined you must use TF loop
    # In both cases it is just a matter of stacking gradients for each Y
    if tf_loop or y.shape.num_elements() is None:
        i = tf.constant(0, dtype=tf.int32)
        y_size = tf.size(y)
        rows = tf.TensorArray(dtype=y.dtype, size=y_size, element_shape=x.shape)
        _, rows = tf.while_loop(
            lambda i, rows: i < y_size,
            lambda i, rows: [i + 1, rows.write(i, tf.gradients(y[i], x)[0])], [i, rows])
        return rows.stack()
    else:
        return tf.stack([tf.gradients(y[i], x)[0] for i in range(y.shape.num_elements())], axis=0)


def hessian_multivar(ys, xs, tf_loop=False):
    # List of list of pieces of the Hessian matrix
    hessian_pieces = [[None] * len(xs) for _ in xs]
    # Hessian with respect to each x (diagonal pieces of the full Hessian)
    for i, h in enumerate(tf.hessians(ys, xs)):
        hessian_pieces[i][i] = h
    # First-order derivatives
    xs_grad = tf.gradients(ys, xs)
    # Pairwise second order derivatives as Jacobian matrices
    for (i1, (x1, g1)), (i2, (x2, g2)) in combinations(zip(count(), zip(xs, xs_grad)), 2):
        # Derivates in both orders
        hessian_pieces[i1][i2] = jacobianhess(g1, x2, tf_loop=tf_loop)
        hessian_pieces[i2][i1] = jacobianhess(g2, x1, tf_loop=tf_loop)
    # Concatenate everything together
    return tf.concat([tf.concat(hp, axis=1) for hp in hessian_pieces], axis=0)


def model_l1_l2_func(nm_set_points, n_in, nn_1, opt_obj, **kwargs_vals):
    hess_approx_flag = False
    neurons_cnt_x1, initializer = kwargs_vals['neurons_cnt'], kwargs_vals['initializer']
    wb_sizes_classif, wb_shapes = kwargs_vals['sizes'], kwargs_vals['shapes']
    x_trained = kwargs_vals['xtr']
    y_trained = kwargs_vals['ytr']
    sess_values = kwargs_vals['sess']
    neurons_cnt = kwargs_vals['neurons_cnt']
    # pcsv = np.genfromtxt('results_paramsparse.csv', delimiter='\t')
    # p = tf.Variable(pcsv, dtype=tf.float64)
    p = tf.Variable(initializer([neurons_cnt], dtype=tf.float64))
    p_store = tf.Variable(tf.zeros([neurons_cnt_x1], dtype=tf.float64))
    save_params_p = tf.assign(p_store, p)
    restore_params_p = tf.assign(p, p_store)
    I_mat = tf.eye(neurons_cnt_x1, dtype=tf.float64)
    shaped_new = np.int(wb_sizes_classif[0]) + np.int(wb_sizes_classif[1])
    # l2_norm_val, all_reg0 = func_structured_l2pen(p, wb_sizes_classif, wb_shapes)
    lambda_param = kwargs_vals['lambda_param']
    lambda_param2 = kwargs_vals['lambda_param2']
    # all_reg_0 = tf.reduce_sum(tf.abs(lasso_p))
    # l2 structured norm loss function
    y_hat_model, y_hat_model_flat_x, y_labeled, x_in, r, l2_norm_val, all_reg0, l2_p, lassop = func_mse_l2(n_in, p, nn_1, kwargs_vals)
    r1 = y_labeled - y_hat_model
    loss_val = tf.reduce_sum(tf.square(r1)) + lambda_param * all_reg0 + lambda_param2 * l2_norm_val
    mu = tf.placeholder(tf.float64, shape=[1]) # LM parameter
    # initialized store for all params, grad and hessian to be trained
    feed_dict = {x_in: x_trained, y_labeled: y_trained}

    if hess_approx_flag:
        jcb = jacobian(y_hat_model, p)
        grads = tf.stack([tf.gradients(yi, p)[0] for yi in tf.unstack(y_hat_model, axis=1)], axis=1)
        print(grads.shape)
        # g_vals = sess_values.run(grads, feed_dict=feed_dict)
        t_jcb = tf.matmul(tf.transpose(jcb), jcb)
        j1 = jacobian_mse(y_hat_model, p, nm_set_points, wb_sizes_classif, wb_shapes)
        jt = tf.transpose(j1)
        partitioned = tf.dynamic_partition(j1, nm_set_points, 1, name='dynamic_unstack')
        print(len(partitioned))
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
        jt_hess = jt[0:shaped_new] + lambda_param2 * l2grad # l2_p_grads, 1)
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
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.squeeze(tf.hessians(loss_val, p)[0])
        jtr = -tf.gradients(loss_val, [p])[0] # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))
        # jtj = hessian_multivar(loss_val, [p])

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x1, neurons_cnt_x1), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x1, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, p)])
    # p2 = p.assign(p + dx)
    sess_values = kwargs_vals['sess']
    
    feed_dict[mu] = np.array([0.01], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss_val, feed_dict)
    
    while feed_dict[mu] > 1e-6 and step < 500:
        p0 = sess_values.run(p)
        p_0_indices = np.where(p == 0)
        p0[p_0_indices] = 0.0
        step += 1
        sess_values.run(save_params_p)
        sess_values.run(restore_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(jtr_store, feed_dict)
        sess_values.run(save_jtj_jtr, feed_dict)
        for _ in range(400):
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(jtj_store, feed_dict)
            sess_values.run(jtr_store, feed_dict)
            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(p)
            p0[np.where(p0 == 0)] = 0
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            new_loss = sess_values.run(loss_val, feed_dict)
            # sess_values.run(save_jtj_jtr, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                # shaped_new = np.int(wb_sizes_classif[0]) + np.int(wb_sizes_classif[1])
                lasso_p0 = p0[shaped_new:]
                in2_hidden_params = p0[0:shaped_new]
                # mat_values.append(lasso_p0)
                mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2]
                    # send the parameters to compute the values of structured penalty after
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    osc_vec0 = np.where((sgn1 < 0.0) & (sgn2 < 0.0))
                    px[osc_vec0] = 0.0
                    # join both sets of parameter lists here
                    px0 = tf.concat([in2_hidden_params, px], 0)
                    
                    if lambda_param2 > 0.0 and np.mod(step, 5) == 0:
                        px0 = sess_values.run(px0)
                        new_all_params, ws_bs_in1_hid1, condvec = func_compute_cond(px0, lambda_param2, kwargs_vals)
                    else:
                        new_all_params = np.array(sess_values.run(px0))
                    p0 = func_collect_allparams(new_all_params, wb_sizes_classif, wb_shapes)
                    p.assign(p0)
                    mat_values = []
                    # mat_values = [px]
                else:
                    p.assign(p0)
                # sess_values.run(jtj_store, feed_dict)
                # sess_values.run(jtr_store, feed_dict)
                # sess_values.run(save_jtj_jtr, feed_dict)
                # sess_values.run(save_params_p)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                p.assign(p0)
                # sess_values.run(save_params_p)
                sess_values.run(restore_params_p)
                # sess_values.run(save_jtj_jtr, feed_dict)
                # sess_values.run(save_params_p)
        if not success:
            print('Failed to improve')
            break
    
    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.01)
    p_new[idx_absp] = 0.0
    new_all_params, ws_bs_in1_hid1, condvec = func_compute_cond(p_new, lambda_param2, kwargs_vals)
    p_new = func_collect_allparams(p_new, wb_sizes_classif, wb_shapes)
    # p_new[osc_vec0]=0.0
    non_zero = np.count_nonzero(p_new)
    y_predict, x_inputs = func_pred_new(n_in, nn_1, p_new, **kwargs_vals)
    inw_hid1 = tf.reshape(p_new[0:shaped_new], shape=(wb_shapes[0][0] + wb_shapes[1][0], wb_shapes[0][1]))
    feed_dict2={x_inputs: x_trained}
    print('ENDED ON STEP: ', ' FINAL LOSS:')
    print(step, current_loss)
    print('Input -> hidden layer 1 Parameters: ')
    print(sess_values.run(inw_hid1))
    # cv.close()
    y_model = sess_values.run(y_predict, feed_dict2)
    return restore_params_p, p_new, y_model, current_loss, non_zero


def model_main_func_l1(nm_set_points, n_in, nn_1, opt_obj, **kwargs_values):
    hess_approx = False
    neurons_cnt_x, sess = kwargs_values['neurons_cnt'], kwargs_values['sess']
    all_sizes, all_shapes = kwargs_values['sizes'], kwargs_values['shapes']
    y_hat, y_hat_flat_x, y, p, x_in = func_mse_loss(n_in, nn_1, **kwargs_values)
    x_tr = kwargs_values['xtr']
    y_tr = kwargs_values['ytr']
    all_reg_0 = tf.reduce_sum(tf.abs(p))
    r = y - y_hat
    lambda1 = 0.001
    loss = tf.reduce_sum(tf.square(r)) + lambda1 * all_reg_0
    mu = tf.placeholder(tf.float64, shape=[1])
    p_store = tf.Variable(tf.zeros([neurons_cnt_x], dtype=tf.float64))
    save_params_p = tf.assign(p_store, p)
    restore_params_p = tf.assign(p, p_store)
    I_mat = tf.eye(neurons_cnt_x, dtype=tf.float64)
    I_mat_diag = tf.eye(neurons_cnt_x, dtype=tf.float64)

    if hess_approx:
        j1, dx_dt = jacobian_mse(y_hat, p, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        jtr = 2 * (tf.matmul(jt, r)) + lambda1 * tf.reshape(dx_dt, shape=(neurons_cnt_x, 1))
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss, p)[0]
        jtr = -tf.gradients(loss, p)[0] # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x, neurons_cnt_x), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)

    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, p)])
    # p2 = p.assign(p + dx)
    sess_values = kwargs_values['sess']
    feed_dict = {x_in: x_tr, y: y_tr}
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
            values_vec = np.where(p0 == 0.0)
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
                    px[p_0_indices] = 0.0
                    print(len(mat_values))
                    p.assign(px)
                    print(p_0_indices)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
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
    abs_p = np.abs(p_new)
    ixd_p = np.where(abs_p < 1e-02)
    p_new[ixd_p] = 0.0
    p20 = p_new
    p20[p_0_indices] = 0.0
    p20 = func_collect_allparams(p20, all_sizes, all_shapes)
    ymodel, x_inputs = func_pred_new(n_in, nn_1, p20, **kwargs_values)
    feed_dict_vals = {x_inputs: x_tr}
    print('ENDED ON STEP: ', ' FINAL LOSS:')
    print(step, current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print('p2:')
    y_model = sess_values.run(ymodel, feed_dict_vals)
    print(p20)
    return restore_params_p, p20, y_model


def jacobian_classif(y, x, m):
    stopgrads = tf.where(x == 0)
    loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(TF_DATA_TYPE, size=m),]
    _, jacobian_classif = tf.while_loop(lambda i, _: i < m, lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x, stop_gradients=stopgrads, unconnected_gradients='zero')[0])), loop_vars)
    dxdt = tf.gradients(tf.reduce_sum(tf.abs(x)), x, unconnected_gradients='zero')[0]
    print(jacobian_classif.stack())
    return jacobian_classif.stack(), dxdt


def func_classifier_l2l1(xtest1, ytest, kwargs1, kwargspred, **kwargs):
    hess_approx_flag = False
    initializer = kwargs1['initializer']
    mu1, _, mu_dec, max_inc = kwargs['mu'], kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    wb_shapes, wb_sizes_classif, hidden = kwargspred['wb_shapes'], kwargspred['wb_sizes'], kwargspred['hidden']
    activation, xydat, ydatrain = kwargspred['activation'], kwargspred['xydat'], kwargspred['xydatrain']
    x_in, nclasses = kwargspred['xtr'], kwargs1['nclasses']
    y_labeled = kwargspred['ytr']
    nm_set_points = x_in.shape[0]
    sess, neurons_cnt_x1 = kwargspred['sess'], kwargspred['neurons_cnt']
    opt_obj = kwargspred['opt_obj']
    params0 = tf.Variable(initializer([neurons_cnt_x1], dtype=tf.float64))
    loss, x, y, y_hat_model , l2_norm_val= func_cross_entropy_loss(wb_sizes_classif, params0, kwargs1)
    feed_dict = {x:x_in, y:y_labeled}
    feed_dict2 = {x: xtest1, y: ytest}
    # check paper and add selected features
    # add correlation for Park data set
    # tuning parameters
    lambda_param = 0.008
    lambda_param2 = 0.4
    # l2 structured norm loss function
    mu = tf.placeholder(tf.float64, shape=[1]) 
    # initialized store for all parameters, gradient and H-matrix to be trained # LM parameter
    p_store = tf.Variable(tf.zeros([neurons_cnt_x1], dtype=tf.float64))
    save_params_p = tf.compat.v1.assign(p_store, params0)
    restore_params_p = tf.compat.v1.assign(params0, p_store)
    I_mat = tf.eye(neurons_cnt_x1, dtype=tf.float64)

    shaped_new = np.int(wb_sizes_classif[0]) + np.int(wb_sizes_classif[1])
    lasso_p = params0[shaped_new:]
    l2_p = params0[0:shaped_new]
    print(lasso_p)
    all_reg0 = tf.reduce_sum(tf.abs(lasso_p))
    loss_val = loss + lambda_param * all_reg0 + lambda_param2 * l2_norm_val

    if hess_approx_flag:
        # j1 equal jacobian_classif(y_hat_model, p, nm_set_points)
        # jt = tf.transpose(j1)
        # jtj = tf.matmul(jt, j1)
        # jtr = tf.matmul(jt, r)
        jcb = jacobian(y_hat_model, params0)
        t_jcb = tf.matmul(tf.transpose(jcb), jcb)
        j1 = jacobian_mse(y_hat_model, params0, nm_set_points, wb_sizes_classif, wb_shapes)
        jt = tf.transpose(j1)
        partitioned = tf.dynamic_partition(j1, nm_set_points, 1, name='dynamic_unstack')
        print(len(partitioned))
        l2_grad = tf.gradients(l2_norm_val, l2_p)[0]
        dxdt = tf.expand_dims(tf.gradients(all_reg0, lasso_p)[0], 1)
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
        jt_hess = jt[0:shaped_new] + lambda_param2 * l2grad # l2_p_grads, 1)
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
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        # jtj = hessian_multivar(loss_val, [params0])
        jtj = tf.hessians(loss_val, params0)[0]
        jtr = -tf.gradients(loss_val, params0)[0] # stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x1, neurons_cnt_x1), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x1, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, params0)])
    # p2 equal p.assign(p + dx)
    sess_values = kwargspred['sess']
    # print(sess_values.run(lasso_p)) 
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss_val, feed_dict)
    
    while feed_dict[mu] > 1e-10 and step < 200:
        p0 = sess_values.run(params0)
        values_vec = np.where(params0 == 0)
        p0[values_vec] = 0.0
        step += 1
        sess.run(save_params_p)
        # sess.run(restore_params_p)
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
            p0 = sess_values.run(params0)
            # p0 equals tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            new_loss = sess_values.run(loss_val, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                shaped_new = np.int(wb_sizes_classif[0]) + np.int(wb_sizes_classif[1])
                lasso_p0 = p0[shaped_new:]
                in2_hidden_params = p0[0:shaped_new]
                mat_values.append(lasso_p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2] # store parameters
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    values_vec = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[values_vec] = 0.0
                    print(len(mat_values))
                    # join both sets of parameter lists here joined_params = np.concatenate(l2_params_set, new_p0)
                    px0 = tf.concat([in2_hidden_params, px], 0)
                    if lambda_param2 > 0.0 and np.mod(step, 2) == 0:
                        px0 = sess_values.run(px0)
                        new_all_params, ws_bs_in1_hid1, _ = func_compute_cond(px0, lambda_param2, kwargspred)
                    else:
                        new_all_params = np.array(sess_values.run(px0))
                    # sess_px0 equal np.array(sess_values.run(px0))
                    p_values_send = func_collect_allparams(new_all_params, wb_sizes_classif, wb_shapes)
                    print(p_values_send.shape)
                    params0.assign(p_values_send)
                    i_cnt = 0
                    mat_values = []
                    mat_values = [px]
                else:
                    params0.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                params0.assign(p0)
            # sess.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break
    
    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.1)
    p_new[idx_absp] = 0.0
    p_new[values_vec] = 0.0
    correct_prediction, feed_dict2, y_hat_classif_logits = predclassif(wb_sizes_classif, xydat, hidden, p_new, activation, wb_shapes, nclasses)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('ENDED ON STEP: ')
    print(step)
    print(' FINAL LOSS:')
    print(current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print("Accuracy:", sess.run(accuracy, feed_dict2))
    correct_predictions = sess.run(y_hat_classif_logits, feed_dict2)
    correct_prediction, feed_dict21, y_hat_classif_logits = predclassif(wb_sizes_classif, ydatrain, hidden, p_new, activation, wb_shapes, nclasses)
    correct_predictions_train = sess.run(y_hat_classif_logits, feed_dict21)
    return p_new, correct_predictions, correct_predictions_train


def func_classifier_l1(xtest1, ytest, kwargs1, kwargspred, **kwargs):
    hess_approx_flag = False
    initializer = kwargs1['initializer']
    mu1, _, mu_dec, max_inc = kwargs['mu'], kwargs['mu_inc'], kwargs['mu_dec'], kwargs['mu_inc']
    wb_shapes, wb_sizes_classif, hidden = kwargspred['wb_shapes'], kwargspred['wb_sizes'], kwargspred['hidden']
    activation, xydat= kwargspred['activation'], kwargspred['xydat']
    x_in = kwargspred['xtr']
    y_labeled, nclasses = kwargspred['ytr'], kwargs1['nclasses']
    sess, neurons_cnt_x1 = kwargspred['sess'], kwargspred['neurons_cnt']
    opt_obj = kwargspred['opt_obj']
    params0 = tf.Variable(initializer([neurons_cnt_x1], dtype=tf.float64))
    loss, x, y, y_hat_model = func_cross_entropy_loss(wb_sizes_classif, params0, kwargs1)
    feed_dict = {x:x_in, y:y_labeled}
    feed_dict2 = {x: xtest1, y: ytest}
    # regularization parameters
    lambda_param = 0.005
    # l2 structured norm loss function
    mu = tf.placeholder(tf.float64, shape=[1]) 
    # initialized store for all params, grad and hessian to be trained # LM parameter
    p_store = tf.Variable(tf.zeros([neurons_cnt_x1], dtype=tf.float64))
    save_params_p = tf.assign(p_store, params0)
    restore_params_p = tf.assign(params0, p_store)
    I_mat = tf.eye(neurons_cnt_x1, dtype=tf.float64)
    all_reg_0 = tf.reduce_sum(tf.abs(params0))
    loss_val = loss + lambda_param * all_reg_0

    if hess_approx_flag:
        j1 = jacobian_classif(y_hat_model, p, nm_set_points)
        jt = tf.transpose(j1)
        jtj = tf.matmul(jt, j1)
        jtr = tf.matmul(jt, r)
    else:
        # remove it
        # stop_grads = tf.where(tf.math.equal(p, 0))
        jtj = tf.hessians(loss_val, params0)[0]
        jtr = -tf.gradients(loss_val, params0)[0] # , stop_gradients=stop_grads, unconnected_gradients='zero')[0]
        jtr = tf.reshape(jtr, shape=(neurons_cnt_x1, 1))

    jtj_store = tf.Variable(tf.zeros((neurons_cnt_x1, neurons_cnt_x1), dtype=tf.float64))
    jtr_store = tf.Variable(tf.zeros((neurons_cnt_x1, 1), dtype=tf.float64))
    save_jtj_jtr = [tf.assign(jtj_store, jtj), tf.assign(jtr_store, jtr)]

    input_mat = jtj_store + tf.multiply(mu, I_mat)
    try:
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    except:
        c = tf.constant(0.1, dtype=tf.float64)
        input_mat += np.identity(input_mat.shape) * c
        dx = tf.matmul(tf.linalg.inv(input_mat, adjoint=None), jtr_store)
    dx = tf.squeeze(dx)
    lm = opt_obj.apply_gradients([(-dx, params0)])
    # p2 = p.assign(p + dx)
    sess_values = kwargspred['sess']
    # print(sess_values.run(lasso_p)) 
    feed_dict[mu] = np.array([0.1], dtype=np.float64)
    i_cnt = 0
    step = 0
    mat_values = []
    sess_values.run(tf.global_variables_initializer())
    current_loss = sess_values.run(loss_val, feed_dict)
    
    while feed_dict[mu] > 1e-10 and step < 200:
        p0 = sess_values.run(params0)
        values_vec = np.where(params0 == 0)
        p0[values_vec] = 0.0
        step += 1
        sess.run(save_params_p)
        # sess.run(restore_params_p)
        if math.log(step, 2).is_integer():
            print('step', 'mu: ', 'current loss: ')
            print(step, feed_dict[mu][0], current_loss)
        success = False
        sess_values.run(jtj_store, feed_dict)
        sess_values.run(p_store)
        for _ in range(300):
            sess_values.run(save_jtj_jtr, feed_dict)
            sess_values.run(jtj_store, feed_dict)
            # p0 equals  session object with run of p2 and feed dict
            sess_values.run(lm, feed_dict)
            p0 = sess_values.run(params0)
            # p0 = tf.where(p == 0, tf.zeros_like(p), p)
            values_vec = np.where(p0 == 0.0)
            p0[values_vec] = 0.0
            new_loss = sess_values.run(loss_val, feed_dict)
            if new_loss < current_loss:
                # divide parameters to 2 groups: 1 for l1 and the other for structured l2
                mat_values.append(p0)
                i_cnt += 1
                if len(mat_values) == 3:
                    sgn1 = mat_values[0] * mat_values[1]
                    sgn2 = mat_values[1] * mat_values[2] # store parameters
                    # checking if parameters are locally close to zero
                    px = mat_values[2]
                    values_vec = np.where((sgn1 < 0) & (sgn2 < 0))
                    px[values_vec] = 0.0
                    print(len(mat_values))
                    # join both sets of parameter lists here joined_params = np.concatenate(l2_params_set, new_p0)
                    p_values_send = func_collect_allparams(px, wb_sizes_classif, wb_shapes)
                    print(p_values_send.shape)
                    params0.assign(p_values_send)
                    i_cnt = 0
                    mat_values = []
                    # mat_values = [px]
                else:
                    params0.assign(p0)
                feed_dict[mu] /= 10
                current_loss = new_loss
                success = True
                break
            else:
                feed_dict[mu] *= 10
                params0.assign(p0)
            # sess.run(save_params_p)
                sess_values.run(restore_params_p)
        if not success:
            print('Failed to improve')
            break
    p_new = sess_values.run(restore_params_p)
    abs_p = np.abs(p_new)
    idx_absp = np.where(abs_p < 0.1)
    p_new[idx_absp] = 0.0
    p_new[values_vec] = 0.0
    correct_prediction, feed_dict2, y_hat_classif_logits = predclassif(wb_sizes_classif, xydat, hidden, p_new, activation, wb_shapes, nclasses)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('ENDED ON STEP: ')
    print(step)
    print(' FINAL LOSS:')
    print(current_loss)
    print('Parameters: ')
    print(sess_values.run(restore_params_p))
    print('Parameters: ')
    print(p_new)
    print("Accuracy:", sess.run(accuracy, feed_dict2))
    correct_predictions = sess.run(y_hat_classif_logits, feed_dict2)
    return p_new, correct_predictions
