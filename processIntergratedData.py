import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class HeartDiseaseData(object):
    """docstring for HeartDiseaseData."""

    def __init__(self):
        super(HeartDiseaseData, self).__init__()
        self.filename_74 = None
        self.filename_14feat = None

    def load_data_file(self, arg_file):
        data74params = pd.read_csv(arg_file, sep=',', header=None)
        self.filename_74 = data74params
        return self.filename_74

    def load_dat_14(self, arg):
        data14feats = pd.read_csv(arg)
        self.filename_14feat = data14feats
        return self.filename_14feat


def main_74b(arg_bigfile):
    # data with 74 features
    h_dat_class = HeartDiseaseData()
    # call to the load_data_file
    dat74params = h_dat_class.load_data_file(arg_bigfile)
    # Data with 14 features
    # dat14feats = h_dat_class.load_dat_14(arg_smallfile)
    return dat74params  # , dat14feats


def main_74(file_74):
    data74feats = pd.read_csv(file_74, sep=',', header='None')
    ydat = data74feats[57]
    return data74feats, ydat


def prep_data_2_train(datafile):
    print(datafile.shape)
    y = datafile[57]
    datafile = datafile.drop(datafile.columns[[0, 57, 75]], axis=1)
    print(datafile.shape)
    min_max_scaling = preprocessing.MinMaxScaler()
    training_x = min_max_scaling.fit_transform(datafile)
    # for eachx in datafile:
    #         datafile[eachx] = (datafile[eachx] - datafile[eachx].min())/datafile[eachx].max()
    # training_x = pd.DataFrame(datafile)
    # training_x = (datafile - datafile.min())/(datafile.max() - datafile.min())
    xfeats = pd.DataFrame(training_x)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(xfeats.corr(), fignum=f.number)
    plt.xticks(range(xfeats.shape[1]), xfeats.columns, fontsize=12, rotation=45)
    plt.yticks(range(xfeats.shape[1]), xfeats.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.show()
    # training_x = training_x.to_numpy()
    y1 = y.to_numpy()
    onehotencoder = OneHotEncoder(categories='auto')
    y2 = onehotencoder.fit_transform(y1.reshape(-1, 1))
    encoder = LabelEncoder()
    # y = encoder.fit_transform(y1.reshape(-1, 1))
    y = np.array([0 if xy < 0.5 else 1 for xy in y1])
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    y_array = y.toarray()
    print(training_x.shape)
    x_train, test_x, y00_train, test_y = train_test_split(training_x, y_array, test_size=0.3)
    # y00_train = y00_train[:, None]
    # test_y = test_y[:, None]
    return x_train, test_x, y00_train, test_y


def dat14feats2train(dat14feats):
    min_max_scaled = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaled.fit_transform(dat14feats)
    df14feat = pd.DataFrame(x_scaled)
    y_label = df14feat[13]
    x14feats = df14feat.drop(df14feat.columns[13], axis=1)
    x_train14f, x_test14, y_train14f, y_test14 = train_test_split(x14feats, y_label, test_size=0.3)
    return x_train14f, x_test14, y_train14f, y_test14


def prep_dat14_feat(dat_file):
    mini_maxi_scale = preprocessing.MinMaxScaler()
    x0_scaled = mini_maxi_scale.fit_transform(dat_file)
    df14 = pd.DataFrame(x0_scaled)
    xt, x0_test, ytr, y0_test = train_test_split(df14)


def func_y(y):
    y1 = []
    y2 = []
    for yi in y:
        if yi < 0.5:
            y1.append(1.0)
            y2.append(0.0)
        else:
            y1.append(0.0)
            y2.append(1.0)
        ynew = np.c_[y1, y2]
    return ynew  # np.array(y1)


def load_solar_data(mdatfile, datmsg):
    mat_contents = sio.loadmat(mdatfile, struct_as_record=False)
    oct_struct = mat_contents[datmsg]
    valdata = oct_struct[0, 0].xyvalues
    x_normed = (valdata - valdata.min(0)) / valdata.ptp(0)
    xdata = x_normed[:, 0:-1]
    ydata = x_normed[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.20, shuffle=False)
    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':
    a_file = '~/forLenovoUbuntu/datfile/heartdisease/Integrated.csv'
    # a_file = '~/LM_code_final/data/heartdisease/Integrated.csv'
    # smallfile_feats = '~/LM_code_final/data/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    # smallfile_feats = '~/Desktop/save_3.11/LM_code_final/data/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    dat74features = main_74b(a_file)
    x_train, x_test, y_train, y_test = prep_data_2_train(dat74features)
    # print(x00_train[0:10, 0:20])
    # y_train = np.array([1 if xy < 0.5 else 0 for xy in y_train])
    # print(x_train[0:10])
    #print(x_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
