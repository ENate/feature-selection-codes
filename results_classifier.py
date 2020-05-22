import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score

sess = tf.Session()

def func_prediction_analysis(predictions_nominal0, y_test):
    predictions_nominal0 = predictions_nominal0[:, 0]
    y_test = y_test[:, 0]
    predictions_nominal = [False if x < 0.5 else True for x in predictions_nominal0]
    print(classification_report(y_test, predictions_nominal, digits=3))
    cm = confusion_matrix(y_test, predictions_nominal)
    cfm = sess.run(tf.math.confusion_matrix(y_test, predictions_nominal, num_classes=2))
    true_negative = cfm[0][0]
    false_positive = cfm[0][1]
    false_negative = cfm[1][0]
    true_positive = cfm[1][1]
    print('Confusion Matrix: \n', cfm, '\n')
    print('True Negative:', true_negative)
    print('False Positive:', false_positive)
    print('False Negative:', false_negative)
    print('True Positive:', true_positive)
    print('Correct Predictions', round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions_nominal0)
    roc_auc = metrics.auc(fpr, tpr)
    gsaved = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    gsaved.savefig('/home/nath/finalResults/classifiers/TFPF1.pdf')
    plt.show()