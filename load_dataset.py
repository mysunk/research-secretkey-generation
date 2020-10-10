import pandas as pd
from sklearn import model_selection
from util import *

def minmax_norm(CSI_data1):
    max_v1 = np.max(CSI_data1)
    min_v1 = np.min(CSI_data1)
    CSI_data1 = (CSI_data1 - min_v1) / (max_v1 - min_v1)
    return (CSI_data1)

def mean_data(CSI_data1, CSI_data2):
    CSI_mean1 = (CSI_data1 + CSI_data2) / 2
    return (CSI_mean1)

def standard_norm(CSI_data1):
    mean_ = np.mean(CSI_data1)
    std_ = np.std(CSI_data1)
    CSI_data1 = (CSI_data1 - mean_) / std_
    return (CSI_data1)

def load_dataset(path, num_datas, step_size):
    train_datas, val_datas, test_datas = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i in range(1, 1 + num_datas):
        CSI_data = pd.read_csv(path + 'gain_' + str(i) + '.csv', header=None)

        # Transpose
        CSI_data = CSI_data.values.T
        # 100개만 남김
        # Min-max normalization
        CSI_data = minmax_norm(CSI_data)
        # CSI_data = standard_norm(CSI_data)

        # Make label
        label = np.ones((CSI_data.shape[0],), dtype=int) * i

        # train-test split
        CSI_data1_tr1, CSI_data1_te, label_tr1, label_te = model_selection.train_test_split(CSI_data, label, test_size=0.2,
                                                                                            shuffle=True, stratify=label)
        CSI_data1_tr, CSI_data1_val, label_tr, label_val = model_selection.train_test_split(CSI_data1_tr1, label_tr1,
                                                                                            test_size=0.2, shuffle=True,
                                                                                            stratify=label_tr1)
        train_datas.append(CSI_data1_tr)
        val_datas.append(CSI_data1_val)
        test_datas.append(CSI_data1_te)

        train_labels.append(label_tr)
        val_labels.append(label_val)
        test_labels.append(label_te)

    # generating original data
    train_datas_noisy, val_datas_noisy, test_datas_noisy = [], [], []
    train_datas_mean, val_datas_mean = [], []
    train_data_label, val_data_label, test_data_label = [], [], []

    for i in range(0,num_datas - 1,step_size):
        CSI_mean1_tr = mean_data(train_datas[i], train_datas[i + 1])
        CSI_mean1_val = mean_data(val_datas[i], val_datas[i + 1])
        # CSI_mean1_te = mean_data(test_datas[i], test_datas[i + 1])

        train_datas_mean.append(CSI_mean1_tr)
        train_datas_mean.append(CSI_mean1_tr)
        val_datas_mean.append(CSI_mean1_val)
        val_datas_mean.append(CSI_mean1_val)

        train_datas_noisy.append(train_datas[i])
        train_datas_noisy.append(train_datas[i + 1])
        val_datas_noisy.append(val_datas[i])
        val_datas_noisy.append(val_datas[i + 1])
        test_datas_noisy.append(test_datas[i])
        test_datas_noisy.append(test_datas[i + 1])

        # make label
        train_data_label.append(train_labels[i])
        train_data_label.append(train_labels[i + 1])
        val_data_label.append(val_labels[i])
        val_data_label.append(val_labels[i + 1])
        test_data_label.append(test_labels[i])
        test_data_label.append(test_labels[i + 1])

    # Define noisy data and original data
    x_train_noisy = np.concatenate(train_datas_noisy, axis=0)
    x_train = np.concatenate(train_datas_mean, axis=0)
    x_valid_noisy = np.concatenate(val_datas_noisy, axis=0)
    x_valid = np.concatenate(val_datas_mean, axis=0)
    x_test = np.concatenate(test_datas_noisy, axis=0)

    train_data_label = np.concatenate(train_data_label, axis=0)
    val_data_label = np.concatenate(val_data_label, axis=0)
    test_data_label = np.concatenate(test_data_label, axis=0)

    return x_train_noisy, x_train, x_valid_noisy, x_valid, x_test, train_data_label, val_data_label, test_data_label