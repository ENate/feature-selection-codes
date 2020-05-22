import os
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
import cProfile
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
__path__=[os.path.dirname(os.path.abspath(__file__))]
from .classifStructuredL2 import main_classif
from .deepNetImpl import NeuralNetwork
from .processDataFiles import ProcessMyData
from .drawingNetsformatting import paramreshape
from .results_classifier import func_prediction_analysis
from .results import func_format_weights, process_ws_bs
from .results_classifier_plots import plot_ROC
from .py_lasso_l2 import model_main_func_l1, model_l1_l2_func
from .processIntergratedData import prep_data_2_train, main_74b, func_y
from .LMAlgorithmImpl import load_solar_data, func_pred, build_network_model, data_to_train0, activation_func


def my_fun_file(choose_flag_0, sess):
    
    if choose_flag_0 == 1:
        # file containing Winsconsin cancer data set
        wdata = '~/forLenovoUbuntu/datfile/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
        # heart_data = '/home/nath/tfCodes/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
        heart_data = '~/Desktop/saved_codes/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
        # Load the data for pre-processing
        x_train_old, y_train_old = ProcessMyData().cancerdatadiagnosislabel(heart_data)
        # Parkinsons data
        pdata = '~/forLenovoUbuntu/datfile/otherdata/datasets/parkinsons.data'
        # Format the data set and return with testing sets
        # x_train_processed, y_train_processed, x_test_processed, y_test_processed =ProcessMyData().processcancerdata(x_train_old, y_train_old)
        # y_train_processed = y_train_processed.values
        x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().wisconsindata(wdata)
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().cancerdatadiagnosislabel(heart_data)
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed, _ = ProcessMyData().func_parkinsons_data(pdata)
        # Load the data with 74 features
        a_file = '~/forLenovoUbuntu/datfile/heartdisease/Integrated.csv'
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().main_74data_disease(a_file)
        dat_74_feats = main_74b(a_file)
        # the parkinsons data set
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed = prep_data_2_train(dat_74_feats)
        park_dat = '~/LM_code_final/data/heartdisease/ReplicatedAcousticFeatures-ParkinsonDatabase.csv'
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().parkinsons_replicated_data(park_dat)
        # load for processed cleveland
        processed_cleveland = '~/Desktop/dec2019/0412_folder/saved_latest/data/processed.cleveland.data'
        x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().func_cleveland(processed_cleveland)
        # call to cervical cancer data
        cervical_data = '~/Desktop/dec2019/0412_folder/saved_latest/data/risk_factors_cervical_cancer.csv'
        # x_train_processed, x_test_processed, y_train_processed, y_test_processed = ProcessMyData().cervicalcancerdata(cervical_data)
        # spect data
        spect_data = '~/Desktop/dec2019/0412_folder/saved_latest/data/SPECT.train'
        # formatting feature set
        # y_train_processed = y_train_processed[:, None]
        # y_test_processed = y_test_processed[:, None]
        n_classes = y_train_processed.shape[1] 
        print(y_train_processed.shape)
        parameters_0, w_b_shapes, w_b_sizes, m_hidden, y_test_prob, y_train_prob = main_classif(x_train_processed, y_train_processed, n_classes, x_test_processed, y_test_processed, sess, choose_flag)
        p_theta = np.asarray(parameters_0)
        func_prediction_analysis(y_test_prob, y_test_processed)
        # plot_ROC(y_train_processed, y_train_prob, y_test_processed, y_test_prob)
    else:
        activation = tf.nn.sigmoid
        data_msg = 'solardatanorm'
        data_msg = 'testData'
        m_lab_file = '~/Desktop/NewFolder2112/SparseNet12ab/testData.mat'
        # m_lab_file='~/Desktop/NewFolder2112/SparseNet12ab/solardatanorm.mat'
        opt = tf.train.GradientDescentOptimizer(learning_rate=1)
        SEED = 40
        lambda1_vec = np.array([0.0005])
        lambda2_vec = np.array([0.004])
        m_hidden = [12, 5]
        initializer = tf.contrib.layers.xavier_initializer()
        x_train_old, x_test_old, y_train_old, y_test = ProcessMyData().load_solar_data(m_lab_file, data_msg)
        n_m, n, x_train_processed, x_test_processed, y_train_processed, ytest = ProcessMyData().data_to_train0(x_train_old, y_train_old, x_test_old, y_test)
        print('The number of inputs is: ', n)
        activation = activation_func()
        w_b_shapes, w_b_sizes, neurons_cnt, nn = build_network_model(n, m_hidden)
        kwargs_dict = {'shapes': w_b_shapes, 'sizes': w_b_sizes, 'neurons_cnt': neurons_cnt, 'xtr': x_train_processed, 'ytr': y_train_processed, 'sess': sess, 
        'm_dec0': 10, 'initializer': initializer, 'activation': activation, 'hidden': nn, 'choose_flag': choose_flag}
        print(w_b_sizes)
        print(w_b_shapes)
        lst_nonzero = [ ]
        lst_nonzero2 = [ ]
        lst_err_l1 = [ ]
        lst_err_l2 = [ ]
        # call prediction function and plot scatter relationships
        # for lambda2 in lambda2_vec:
        # for lambda1 in lambda1_vec:
        kwargs_dict['lambda_param'] = lambda1_vec
        kwargs_dict['lambda_param2'] = lambda2_vec
        restore_parameters, p2, y_model_3, current_loss, non_zero = model_l1_l2_func(n_m, n, nn, opt, **kwargs_dict)
        lst_err_l1.append(current_loss)
        lst_nonzero.append(non_zero)
        lst_err_l2.append(lst_err_l1)
        lst_nonzero2.append(lst_nonzero)
        print(lst_err_l2)
        print(lst_nonzero2)
        #
        nm_train = x_train_processed.shape[0]
        y_hat_flat, x = func_pred(n, nn, restore_parameters, **kwargs_dict)
        y_labeled = sess.run(y_hat_flat, feed_dict={x: x_train_processed})
        f1 = plt.figure(1)
        colors = np.random.rand(nm_train)
        area = (10 * np.random.rand(nm_train)) ** 2
        plt.scatter(y_labeled, y_train_processed, s=area, c=colors, alpha=0.5)
        f1.suptitle('Predicting Artificial Data from Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Data', fontsize=14, fontweight='bold')
        f1.savefig('~/finalResults/classifiers/scatter_testData.pdf')
        # test set ##########################################################
        # ####################################################################
        nm_test = x_test_processed.shape[0]
        colors_t = np.random.rand(nm_test)
        area_t = (10 * np.random.rand(nm_test)) ** 2
        g1 = plt.figure(2)
        y_hf1, x = func_pred(n, nn, p2, **kwargs_dict)
        with open("results_paramsparse.csv", "w") as psp:
            writer_p = csv.writer(psp, delimiter='\t',lineterminator='\n',)
            writer_p.writerow(p2)
        yl = sess.run(y_hf1, feed_dict={x: x_test_processed})
        plt.scatter(yl, ytest, s=area_t, c=colors_t, alpha=0.5)
        g1.suptitle('Prediction relationship for Test Set', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Data', fontsize=14, fontweight='bold')
        g1.savefig('~/finalResults/classifiers/scatter_testDatatestset2.pdf')
        # ###################################################################
        g2 = plt.figure(3)
        plt.scatter(y_model_3, y_train_processed, s=area, c=colors, alpha=0.5)
        g2.suptitle('Predicting Artificial Data from Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Data', fontsize=14, fontweight='bold')
        g2.savefig('~/finalResults/classifiers/scatter_testData3.pdf')
        # ##################################################################
        p_theta = np.asarray(p2)
        ws, bs = func_format_weights(p_theta, w_b_sizes, w_b_shapes)
        lst_wsbs = process_ws_bs(ws, bs)
        # figx = go.Figure()
        # figx.add_trace(go.Scatter(x=y_model_3, y=y_train_processed, mode='markers', name='markers'))
        # figx.show()
        # plt.show()
    # Call class to draw network here
    all_net_draw_mat = paramreshape(p_theta, w_b_shapes, w_b_sizes, m_hidden)
    network = NeuralNetwork()
    # loop via formatted matrix as layers in network
    for idx_params in all_net_draw_mat:
        network.add_layer(idx_params.shape[1], idx_params)
        print(idx_params.shape)
    # last layer to output
    nh = np.ones((1, all_net_draw_mat[-1].shape[0]))
    print(nh.shape)
    network.add_layer(nh.shape[1], nh)
    network.draw()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    sess = tf.Session()
    start_time = datetime.now()
    choose_flag = 2
    my_fun_file(choose_flag, sess)
    # cProfile.run('my_fun_file(choose_flag, sess)')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    sess.close()
