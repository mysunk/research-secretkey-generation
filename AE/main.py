from __future__ import print_function
import argparse
import torch.utils.data
from tqdm import *
from AE.load_dataset import *

# modules
from AE.model import *
from AE.util import *


def train(model, epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    list_train = []
    # list_train_loss = []
    # list_valid_loss = []

    patiences = 100
    delta = 0.001
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patiences, verbose=True, delta=delta)

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

            # # early stopping
            # early_stopping(loss, model)
            # if early_stopping.early_stop:
            #    print("Earlystopping")
            #    break

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))
        lr = optimizer.param_groups[0]['lr']
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' + f'learning_rate : {lr}')
        print(print_msg)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # step learning rate
        scheduler.step()

    # loss plot
    plt.figure()
    plt.plot(avg_train_losses, label='Train')
    plt.plot(avg_valid_losses, label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_dir + '/train.png')
    plt.show()

    # reconstruction 확인
    plt.figure()
    plt.plot(data.cpu().detach().numpy().T, 'k')
    plt.plot(recon_batch.cpu().detach().numpy().T, 'b:')
    plt.savefig(save_dir + '/reconstruction.png')
    plt.show()

    # save model
    torch.save({
        'epoch': args.epochs,
        'batch_size': args.batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'step_size': args.step_size,
        'data_path':path
    }, save_dir + '/checkpoint.pt')

    return (model, list_train, train_loss, valid_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    parser.add_argument("--step_size", default=1, type = int)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_step_size", default=10, type=int)
    parser.add_argument("--num_datas", default=4, type=int)
    parser.add_argument("--save_dir", default='results')
    parser.add_argument("--path", default='data/Experiment_0924_gain/')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-f')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    import os
    # Load data
    # path = F"data/Experiment_0924_gain/"
    path = args.path

    # model save dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_train_noisy, x_train, x_valid_noisy, x_valid, x_test, train_data_label, val_data_label, test_data_label = load_dataset(path, num_datas = args.num_datas, step_size = args.step_size)
    len_ = x_train.shape[1]
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
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
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

    model = VAE(len_).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma = args.lr_decay_rate)

    #train
    model, z_coded, train_loss, valid_loss = train(model, args.epochs)
