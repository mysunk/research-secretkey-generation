from model import *
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import *

#%% Input
num_workers = 1
model_save_dir = 'results/0929/7/checkpoint.pt'
len_ = 52

#%% main
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DEVICE = device

    # load model
    model = VAE(len_).to(device)
    checkpoint = torch.load(model_save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    batch_size = checkpoint['batch_size']
    step_size = checkpoint['step_size']
    path = checkpoint['data_path']

    # load dataset
    _, _, _, _, x_test, _, _, test_data_label = load_dataset(path, num_datas=4, step_size = step_size)

    num_samples = len(np.unique(test_data_label))
    test_set = torch.tensor(x_test)

    import torch
    from torch.utils.data import DataLoader
    kwargs = {'num_workers': num_workers, 'pin_memory': True}

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    print('=== End of load dataset ===')

    # test
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

    # coding result
    z_pred = [test_pred_element.cpu().detach().numpy() for test_pred_element in list_test]
    z_pred = np.concatenate(z_pred, axis=0)

    # Evaluation
    def hamming_distance(x, y):
        len_bit = max(len(x), len(y))
        bin_x = x
        bin_y = y

        result = 0
        for i in range(len_bit):
            if bin_x[i] != bin_y[i]:
                result += 1
        return result

    dists = np.zeros((100, num_samples), dtype=int)
    for i, l in enumerate(np.unique(test_data_label)):
        data_1 = z_pred[test_data_label == 1, :]
        data_2 = z_pred[test_data_label == l, :]
        # random sampling
        max_i = data_1.shape[0]
        for j in range(100):
            dists[j, i] = hamming_distance(data_1[np.random.randint(0, max_i, 1)[0],:], data_2[np.random.randint(0, max_i, 1)[0],:])

    plt.figure(figsize=(10,5))
    plt.boxplot(x=dists)
    # plt.xticks(rotation=30)
    plt.show()

    # reconstruction plot
    plt.figure()
    plt.plot(recon_batch.cpu().numpy().T)
    plt.show()

    plt.figure()
    plt.boxplot(x = z_pred[test_data_label == 1, :].T)
    plt.show()