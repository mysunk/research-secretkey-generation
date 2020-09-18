from __future__ import print_function
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
#Check model structure
from torchsummary import summary
from tqdm import *

# modules
from model import *
from util import *

#%%
def train(model, batch_size, epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    list_train = []

    list_train_loss = []
    list_valid_loss = []

    patiences = 100
    delta = 0.001
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patiences, verbose=True, delta=deltas)

    for epoch in range(1, epochs + 1):
        "train the model"

        model.train()
        for batch, sample_data in tqdm(enumerate(train_loader, 1)):
            data = sample_data[0].to(DEVICE)
            data_noisy = sample_data[1].to(DEVICE)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            recon_batch, z_coded, mu, logvar, z = model(data_noisy.float())
            # calculate the loss
            loss = loss_function(recon_batch, data, mu, logvar, z, z_coded)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            list_train.append(z_coded)

        "validate the model"
        model.eval()
        for batch, sample_data in tqdm(enumerate(valid_loader, 1)):
            data = sample_data[0].to(DEVICE)
            data_noisy = sample_data[1].to(DEVICE)

            recon_batch, z_coded, mu, logvar, z = model(data_noisy.float())
            loss = loss_function(recon_batch, data, mu, logvar, z, z_coded)
            valid_losses.append(loss.item())
            # early_stopping(loss, model)
            # if early_stopping.early_stop:
            #    print("Earlystopping")
            #    break

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping(valid_loss, model)
        # if early_stopping.early_stop:
        #    print("Early stopping")
        #    break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return (model, list_train, train_losses, valid_losses)

def test(epoch):
    test_losses = []
    list_test = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(DEVICE)
            recon_batch, z_coded, mu, logvar, z = model(data.float())
            test_loss += loss_function(recon_batch, data, mu, logvar, z, z_coded).item()
            test_losses.append(test_loss)
            # print(i)
            list_test.append(z_coded)

    test_loss /= (i + 1)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return (list_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-f')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    def normalize_convert(CSI_data1):
        max_v1 = np.max(CSI_data1)
        min_v1 = np.min(CSI_data1)
        CSI_data1 = (CSI_data1 - min_v1) / (max_v1 - min_v1)
        return (CSI_data1)


    def mean_data(CSI_data1, CSI_data2):
        CSI_mean1 = (CSI_data1 + CSI_data2) / 2
        return (CSI_mean1)


    # Load data
    path1 = F"data/sanitized_phase/sanitized_phase_1.csv"
    path2 = F"data/sanitized_phase/sanitized_phase_2.csv"
    path3 = F"data/sanitized_phase/sanitized_phase_3.csv"
    path4 = F"data/sanitized_phase/sanitized_phase_4.csv"

    CSI_data1 = pd.read_csv(path1, header=None)
    CSI_data2 = pd.read_csv(path2, header=None)
    CSI_data3 = pd.read_csv(path3, header=None)
    CSI_data4 = pd.read_csv(path4, header=None)

    # Transpose
    CSI_data1 = CSI_data1.values.T
    CSI_data2 = CSI_data2.values.T
    CSI_data3 = CSI_data3.values.T
    CSI_data4 = CSI_data4.values.T

    # Normalization, generating Mean_data
    CSI_data1 = normalize_convert(CSI_data1)
    CSI_data2 = normalize_convert(CSI_data2)
    CSI_data3 = normalize_convert(CSI_data3)
    CSI_data4 = normalize_convert(CSI_data4)

    from sklearn import model_selection

    CSI_data1_tr1, CSI_data1_te = model_selection.train_test_split(CSI_data1, test_size=0.2, shuffle=None)
    CSI_data2_tr1, CSI_data2_te = model_selection.train_test_split(CSI_data2, test_size=0.2, shuffle=None)
    CSI_data3_tr1, CSI_data3_te = model_selection.train_test_split(CSI_data3, test_size=0.2, shuffle=None)
    CSI_data4_tr1, CSI_data4_te = model_selection.train_test_split(CSI_data4, test_size=0.2, shuffle=None)

    CSI_data1_tr, CSI_data1_val = model_selection.train_test_split(CSI_data1_tr1, test_size=0.2, shuffle=None)
    CSI_data2_tr, CSI_data2_val = model_selection.train_test_split(CSI_data2_tr1, test_size=0.2, shuffle=None)
    CSI_data3_tr, CSI_data3_val = model_selection.train_test_split(CSI_data3_tr1, test_size=0.2, shuffle=None)
    CSI_data4_tr, CSI_data4_val = model_selection.train_test_split(CSI_data4_tr1, test_size=0.2, shuffle=None)

    CSI_mean1_tr = mean_data(CSI_data1_tr, CSI_data2_tr)
    CSI_mean1_val = mean_data(CSI_data1_val, CSI_data2_val)
    CSI_mean1_te = mean_data(CSI_data1_te, CSI_data2_te)

    CSI_mean2_tr = mean_data(CSI_data3_tr, CSI_data4_tr)
    CSI_mean2_val = mean_data(CSI_data3_val, CSI_data4_val)
    CSI_mean2_te = mean_data(CSI_data3_te, CSI_data4_te)

    # %%
    # x_train_noisy = np.concatenate([CSI_data1_tr, CSI_data2_tr, CSI_data3_tr, CSI_data4_tr], axis=0)
    # x_train = np.concatenate([CSI_mean1_tr, CSI_mean1_tr, CSI_mean2_tr, CSI_mean2_tr], axis=0)
    # x_valid_noisy = np.concatenate([CSI_data1_val, CSI_data2_val, CSI_data3_val, CSI_data4_val], axis=0)
    # x_valid = np.concatenate([CSI_mean1_val, CSI_mean1_val, CSI_mean2_val, CSI_mean2_val], axis=0)
    # x_test = np.concatenate([CSI_data1_te, CSI_data2_te, CSI_data3_te, CSI_data4_te], axis=0)

    #    x_train_noisy = np.concatenate([CSI_data1_tr, CSI_data2_tr, CSI_data3_tr, CSI_data4_tr], axis=0)
    #    x_train = np.concatenate([CSI_mean1_tr, CSI_mean1_tr, CSI_mean2_tr, CSI_mean2_tr], axis=0)
    #    x_valid_noisy = np.concatenate([CSI_data1_val, CSI_data2_val, CSI_data3_val, CSI_data4_val], axis=0)
    #    x_valid = np.concatenate([CSI_mean1_val, CSI_mean1_val, CSI_mean2_val, CSI_mean2_val], axis=0)
    #    x_test = np.concatenate([CSI_data1_te, CSI_data2_te, CSI_data3_te, CSI_data4_te], axis=0)

    x_train_noisy = np.concatenate([CSI_data1_tr, CSI_data2_tr], axis=0)
    x_train = np.concatenate([CSI_mean1_tr, CSI_mean1_tr], axis=0)
    x_valid_noisy = np.concatenate([CSI_data1_val, CSI_data2_val], axis=0)
    x_valid = np.concatenate([CSI_mean1_val, CSI_mean1_val], axis=0)
    x_test = np.concatenate([CSI_data1_te, CSI_data2_te], axis=0)

    # x_train1, x_valid = model_selection.train_test_split(input_data, test_size=0.2, shuffle=None)
    # x_train_noisy1, x_valid_noisy = model_selection.train_test_split(input_data_noisy, tet_size=0.2, shuffle=None)

    train_set = torch.tensor(x_train)
    train_noisy_set = torch.tensor(x_train_noisy)
    valid_set = torch.tensor(x_valid)
    valid_noisy_set = torch.tensor(x_valid_noisy)
    test_set = torch.tensor(x_test)

    from torch.utils.data import TensorDataset

    train_data = TensorDataset(train_set, train_noisy_set)
    valid_data = TensorDataset(valid_set, valid_noisy_set)

    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DEVICE = device
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)
    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             **kwargs)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # summary(model, (64, 1600))

    model, z_code_tr, train_loss, valid_loss = train(model, args.batch_size, args.epochs)
    z_code_te = test(args.epochs)
