'''
Created on Jul 8, 2019
@author: messa
'''
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing.data import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class ProcessMyData(object):
    def __init__(self):
        self.datafilename = None
    # Import the first data for pre-processing

    def load_solar_data(self, mdatfile, datmsg):
        mat_contents = sio.loadmat(mdatfile, struct_as_record=False)
        oct_struct = mat_contents[datmsg]
        if datmsg == 'testData':
            valdata = oct_struct[0, 0].xyvalues
            x_normed = (valdata - valdata.min(0)) / valdata.ptp(0)
        elif datmsg == 'solardatanorm':
            x_normed = oct_struct[0, 0].xyvalues
        else:
            valdata = oct_struct[0, 0].values
            x_normed = (valdata - valdata.min(0)) / valdata.ptp(0)
        xdata = x_normed[:, 0:-1]
        ydata = x_normed[:, -1]
        self.xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.30, shuffle=False)
        return self.xtrain, xtest, ytrain, ytest

    def data_to_train0(self, xtrain0, ytr0, xt0):
        xtrain00 = np.asanyarray(xtrain0)
        biases = np.ones((xtrain00.shape[0], 1))
        xtr = np.c_[xtrain00, biases]
        #xtr = xtrain00
        # For the sake of testing
        xt00 = np.asanyarray(xt0)
        biases1 = np.ones((xt00.shape[0], 1))
        xt1 = np.c_[xt00, biases1]
        #xt1 = xt00
        ytr = ytr0[:, None]
        n = xtr.shape[1]
        Nm = xtr.shape[0]
        return Nm, n, xtr, xt1, ytr

    def func_parkinsons_data(self, arg_filename):
        all_data = pd.read_csv(arg_filename, sep=',')
        ylabel = all_data['status']
        xfeatures = all_data.drop(['status', 'name'], axis=1)
        xfeats = pd.DataFrame(xfeatures)
        f = plt.figure(figsize=(19, 15))
        plt.matshow(xfeats.corr(), fignum=f.number)
        plt.xticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14, rotation=45)
        plt.yticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        # plt.title('Correlation Matrix', fontsize=16)
        plt.show()
        # print(xfeatures.head())
        for eachx in xfeatures:
            xfeatures[eachx] = (xfeatures[eachx] - xfeatures[eachx].min())/xfeatures[eachx].max()
        ydata = ylabel.to_numpy()
        ydata = ydata[:, None]
        xdata = xfeatures.to_numpy()
        xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.20, shuffle=False)
        #xtrain = np.c_[xtrain, np.ones((xtrain.shape[0], 1))]
        #xtest = np.c_[xtest, np.ones((xtest.shape[0], 1))]
        return xtrain, xtest, ytrain, ytest, all_data
    
    def parkinsons_replicated_data(self, park_dat):
        df_parkinson = pd.read_csv(park_dat, sep=',')
        ylabel = df_parkinson['Status']
        xfeatures = df_parkinson.drop(['Status', 'ID'], axis=1)
        xfeats = df_parkinson.drop(['Status', 'ID'], axis=1).values
        x = (xfeats - np.min(xfeats))/(np.max(xfeats) - np.min(xfeats))
        y = df_parkinson['Status'].values
        xfeatsp = pd.DataFrame(xfeatures)
        minmax_scaling = MinMaxScaler()
        x_scaledp = minmax_scaling.fit_transform(xfeatsp)
        x_scaledp = pd.DataFrame(x_scaledp)
        f1 = plt.figure(figsize=(19, 16))
        plt.matshow(x_scaledp.corr(), fignum=f1.number)
        plt.xticks(range(x_scaledp.shape[1]), x_scaledp.columns, fontsize=10, rotation=45)
        plt.xticks(range(x_scaledp.shape[1]), x_scaledp.columns, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=12)
        plt.show()
        for eachx in xfeatures:
            xfeatures[eachx] = (xfeatures[eachx] - xfeatures[eachx].min())/xfeatures[eachx].max()
        ylabel = ylabel.values
        # ydata = ylabel[:, None]
        xdata = x_scaledp.to_numpy()
        targets = np.array(ylabel).reshape(-1)
        y = np.eye(2)[targets]
        xtrain, xtest, y_train, y_test = train_test_split(x, y, test_size=0.30)#, shuffle=False)
        print(y_test)
        #y_train = ytrain[:, None]
        #y_test = ytest[:, None]
        return xtrain, xtest, y_train, y_test
    
    
    def spectf_data(self, file_named):
        pass
    
    
    def func_cleveland(self, cleve_data):
        df_cleveland = pd.read_csv(cleve_data, sep=',', header=None)
        df_cleveland2 = df_cleveland.replace("?", np.nan)
        df_cleveland2 = df_cleveland2.fillna(df_cleveland2.mean(), inplace=False)
        df_cleveland2.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df_cleveland2['target'] = df_cleveland2.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
        #df_cleveland2['thal'] = pd.to_numeric(df_cleveland2['thal'])
        #my_mean = df_cleveland2['thal'].dropna().mean()
        #df_cleveland2['thal'].fillna(my_mean)
        #df_cleveland2['ca'] = pd.to_numeric(df_cleveland2['ca'])
        #my_mean2 = df_cleveland2['ca'].dropna().mean()
        #df_cleveland2['ca'].fillna(my_mean2)
        feature_mean1 = round(df_cleveland2['ca'].apply(pd.to_numeric).mean(), 1)
        feature_mean2 = round(df_cleveland2['thal'].apply(pd.to_numeric).mean(), 1)
        df_cleveland2['ca'] = df_cleveland2['ca'].fillna(feature_mean1)
        df_cleveland2['thal'] = df_cleveland2['thal'].fillna(feature_mean2)
        X = df_cleveland2.iloc[:, :-3].values
        # y = df_cleveland2.iloc[:, -1].values
        X = (X - np.min(X))/(np.max(X) - np.min(X))
        y = df_cleveland2['target'].to_numpy()
        y[y < 0.5] = 0
        y[y >= 0.5] = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        #min_max_scaling = preprocessing.MinMaxScaler()
        #X_train = min_max_scaling.fit_transform(X_train)
        #X_test = min_max_scaling.fit_transform(X_test)
        y_train = y_train[:, None]
        y_test = y_test[:, None]
        return X_train, X_test, y_train, y_test 
        
    def func_cervical_data(self, cervical_file):
        # read the data set from the file into a pandas frame
        df_cervical = pd.read_csv(cervical_file, sep=';')
        
        
        
    def heartdiseasedata(self, filenameobj):
        # Load the data file
        df = pd.read_csv(filenameobj, sep=",")
        listcols = ['Unnamed: 32', 'id', 'diagnosis']
        x = np.array(df.drop(listcols, axis=1))
        Y = np.array(df.diagnosis)
        return x, Y

        y0 = df.diagnosis.values
        x_data = df.drop(['diagnosis'], axis=1)
        # data array is the xvalues # Extract values as nparray,y dot of values
        classes = df.diagnosis
        encoder = LabelEncoder()
        y0 = encoder.fit_transform(classes)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y0, test_size=0.15, random_state=42)
        return x_train, y_train

    def tffilereader(self, filenamed):
        # Split dataset into separate points (as strings)
        string_points = filenamed.read().split('\n')
        string_points.pop(-1)
        np.random.shuffle(string_points)  # Randomize (avoid bias)
        num_input_features = 30
        # Initialize dataset class arrays
        point_array = np.empty((0, num_input_features))
        y_labels = np.empty((0, 2))

        # Trim data points
        # Format as np.arrays and add to class arrays (Benign or Malignant)
        for point in string_points:
            point = point.split(',')
            if 'M' in point:  # if malignant
                y_labels = np.append(y_labels, [[0, 1]], axis=0)
            else:  # if benign (can only be labeled 'B' or 'M')
                y_labels = np.append(y_labels, [[1, 0]], axis=0)
            point = point[2:]  # trim for only the 10 important features
            temp = np.array(point)  # convert to numpy array
            temp = temp.astype(float)  # cast as float array
            point_array = np.append(point_array, [temp], axis=0)

        # Split training and testing data
        experience_matrix = point_array[0:400]
        experience_matrix_y = y_labels[0:400]
        testmatrix = point_array[400:]
        testmatrixy = y_labels[400:]
        return experience_matrix, experience_matrix_y, testmatrix, testmatrixy

    def cancerdatadiagnosislabel(self, filedatname):
        dataxy = pd.read_csv(filedatname, sep=",", dtype={"diagnosis": "category"})
        # #######################################################################
        dummies = pd.get_dummies(dataxy['diagnosis'], prefix='diagnosis', drop_first=False)
        dataxynew = pd.concat([dataxy, dummies], axis=1)
        dataxynew1 = dataxynew.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
        ylabels = dataxynew1[['diagnosis_B', 'diagnosis_M']]
        xinputs = dataxynew1.drop(['diagnosis_B', 'diagnosis_M'], axis=1)

        for eachx in xinputs:
            xinputs[eachx] = (xinputs[eachx] - xinputs[eachx].min())/xinputs[eachx].max()
        return xinputs, ylabels

    def processcancerdata(self, xinputs, ylabels):
        # features, ylabels = np.array(xinputs), np.array(ylabels)
        features, ylabels = xinputs.to_numpy(), ylabels.to_numpy()
        # features = np.c_[features, np.ones((features.shape[0], 1))]
        # fraction of examples to keep for training
        split_frac = 0.8
        n_records = len(features)
        split_idx = int(split_frac*n_records)
        train_x, train_y = features[:split_idx], ylabels[:split_idx]
        test_x, test_y = features[split_idx:], ylabels[split_idx:]
        return train_x, train_y, test_x, test_y

    def gettheclassifdata(self, file_name_d):
        train_data = pd.read_csv(file_name_d, sep=',')
        # train_columns = list(train_data.columns.values)
        _ = train_data.drop(['id', 'area_mean', 'adiagnosis'], axis=1)
        # r_features = r_data.columns
        train_data.loc[train_data.diagnosis == "M", 'diagnosis'] = 1
        train_data.loc[train_data.diagnosis == "B", 'diagnosis'] = 0
        train_data.loc[train_data.diagnosis == 0, 'benign'] = 1
        train_data.loc[train_data.diagnosis == 1, 'benign'] = 0
        train_data['benign'] = train_data.benign.astype(int)
        train_data = train_data.rename(columns={'diagnosis': 'malignant'})
        Malignant = train_data[train_data.malignant == 1]
        Benign = train_data[train_data.benign == 1]
        train_X = Malignant.sample(frac=0.8)
        # count_Malignants = len(train_X)
        train_X = pd.concat([train_X, Benign.sample(frac=0.8)], axis=0)
        test_X = train_data.loc[~train_data.index.isin(train_X.index)]
        # train_X = shuffle(train_X)
        # test_X = shuffle(test_X)
        train_Y = train_X.malignant
        train_Y = pd.concat([train_Y, train_X.benign], axis=1)
        test_Y = test_X.malignant
        test_Y = pd.concat([test_Y, test_X.benign], axis=1)
        train_X = train_X.drop(['malignant', 'benign'], axis=1)
        test_X = test_X.drop(['malignant', 'benign'], axis=1)
        features = train_X.columns.values
        for feature in features:
            mean, std = train_data[feature].mean(), train_data[feature].std()
            train_X.loc[:, feature] = (train_X[feature] - mean) / std
            test_X.loc[:, feature] = (test_X[feature] - mean) / std
        return train_X, train_Y

    def classifydatalifesciences(self, classifdatafile):
        datxy = pd.read_csv(classifdatafile, sep=",")
        return datxy

    def xtraintestdata(self, datarray, yarray, dfiletowrite):
        x_train, x_test, y_train, y_test = train_test_split(datarray, yarray, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
        min_max_scaler = MinMaxScaler()
        # feed in a numpy array
        x_train_norm = min_max_scaler.fit_transform(x_train)
        _ = np.c_[x_train_norm, y_train]
        dirme = dfiletowrite
        sio.savemat(dirme, mdict={'UCIDat': yarray})
        xy_valid = np.c_[x_val, y_val]
        xy_train = np.c_[x_train, y_train]
        xy_test = np.c_[x_test, y_test]
        return xy_train, xy_test, xy_valid

    def wisconsindata(self, wsdata):
        data = pd.read_csv(wsdata, sep=',')
        data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
        y = data.diagnosis.values
        x_data = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values
        xfeatures = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values
        xfeats = pd.DataFrame(xfeatures)
        f = plt.figure(figsize=(19, 15))
        plt.matshow(xfeats.corr(), fignum=f.number)
        plt.xticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14, rotation=45)
        plt.yticks(range(xfeats.shape[1]), xfeats.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.show()
        x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        y_train = y_train[:, None]
        y_test = y_test[:, None]
        return x_train, x_test, y_train, y_test

    def cancerdatadiagnosislabel1(self, filedatname):
        dataxy = pd.read_csv(filedatname, sep=",", dtype={"diagnosis": "category"})
        dataxynew = dataxy
        dataxynew.diagnosis = [1 if eachv == "M" else 0 for eachv in dataxynew.diagnosis]
        xinputs = dataxynew.drop(['Unnamed: 32', 'id', 'diagnosis'],axis = 1)
        ylabels = dataxynew['diagnosis']

        for eachx in xinputs:
            xinputs[eachx] = (xinputs[eachx] - xinputs[eachx].min())/xinputs[eachx].max()
        x_train, x_test, y_train, y_test = train_test_split(xinputs, ylabels, test_size=0.20, random_state=42)
        y_train = y_train[:, None]
        y_test = y_test[:, None]
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        return x_train, x_test, y_train, y_test

    def cervicalcancerdata(self, cervfilename):
        datcerv = pd.read_csv(cervfilename, sep=',')
        return datcerv

    def heartdata74attr(filename_dat):
        datwith74attributes = pd.read_csv('/home/nath/forLenovoUbuntu/datfileheartdisease')
        return datwith74attributes

    def xread_csv(self, batch_size, file_name, record_defaults):
        filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(value, record_defaults=record_defaults)
        return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)


if __name__ == '__main__':
    wdat = 'datafiles/breast-cancer-wisconsin-data/data.csv'
    pdata = '/home/nath/forLenovoUbuntu/datfile/otherdata/datasets/parkinsons.data'
    heartdata = '/home/nath/datafiles/wiscosincategoricaldata'
    dataxy = '/home/nath/tfCodes/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    mlabfile = '~/JanMatlab/SparseNet12ab/testData.mat'
    riskfactorscervicalcancer = '~/tfExample/datafiles/risk_factors_cervical_cancer.csv'
    datfiletowrite = '~/projectphd/selectedwinfiles/alltests/NewFolder2112/SparseNet12ab/UCIDat.mat'
    wsconsindat = '~/forLenovoUbuntu/datfile/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    p_objs = ProcessMyData()
    hdata = '~/forLenovoUbuntu/datfile/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv'
    x_train, x_test, y_train, y_test = p_objs.wisconsindata(wsconsindat)
    x_train, y_train = p_objs.cancerdatadiagnosislabel(hdata)
    _,_,_,_,_ = p_objs.func_parkinsons_data(pdata)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    park_dat = '~/LM_code_final/data/heartdisease/ReplicatedAcousticFeatures-ParkinsonDatabase.csv'
    p_objs.parkinsons_replicated_data(park_dat)
