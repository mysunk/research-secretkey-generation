import pandas as pd
import numpy as np
from genome import *
import matplotlib.pyplot as plt
import argparse

# load all dataset
CSI_datas = []
num_samples = 40
for i in range(1, num_samples + 1):
    CSI_data = pd.read_csv('data_in_use/gain_' + str(i) + '.csv', header=None)
    # Transpose
    CSI_data = CSI_data.values.T
    # Min-max normalization
    CSI_data = minmax_norm(CSI_data)
    CSI_datas.append(CSI_data)

def minmax_norm(CSI_data1):
    max_v1 = np.max(CSI_data1)
    min_v1 = np.min(CSI_data1)
    CSI_data1 = (CSI_data1 - min_v1) / (max_v1 - min_v1)
    return (CSI_data1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_save_dir", default='1015_11/9', type=str)
    parser.add_argument("--reference", default='gain_20', type=str)
    args = parser.parse_args()
    result_save_dir = 'results/' + args.result_save_dir

    # load reference dataset
    CSI_data_ref = pd.read_csv('data_in_use/' + args.reference + '.csv', header=None)
    CSI_data_ref = CSI_data_ref.values.T
    CSI_data_ref = minmax_norm(CSI_data_ref)

    import pickle
    with open(result_save_dir + '/best_genomes.pkl','rb') as f:
        best_genomes = pickle.load(f)

    #%% check hamming distance
    # predict
    codewords_ref = best_genomes[0].predict(CSI_data_ref)
    # plot
    dists = np.zeros((1000, len(CSI_datas)), dtype=int)
    dists_same_loc = np.zeros((1000, len(CSI_datas)), dtype=int)
    # random sampling
    max_i = CSI_data_ref.shape[0]
    codewords = []
    for j in range(len(CSI_datas)):
        codewords_2 = best_genomes[0].predict(CSI_datas[j])
        codewords.append(codewords_2)
        for i in range(1000):
            # for same sample
            dists[i, j] = hamming_distance(codewords_ref[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
            dists_same_loc[i,j] = hamming_distance(codewords_2[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
    codewords = np.concatenate(codewords, axis=0)

    locs = np.arange(0, 0.06*num_samples, 0.06)

    plt.figure(figsize=(10,5))
    plt.boxplot(x=dists)
    plt.xticks(range(num_samples)[::4], locs[::4])
    plt.ylabel('Hamming distance')
    # plt.xlabel('Sample location #')
    plt.xlabel('d / lambda')
    plt.tight_layout()
    # plt.savefig(result_save_dir + '/hamming_dist.png')
    # plt.show()

    plt.figure(figsize=(10,5))
    plt.boxplot(x=dists_same_loc)
    plt.xticks(range(num_samples)[::4], locs[::4])
    plt.ylabel('Hamming distance')
    # plt.xlabel('Sample location #')
    plt.xlabel('d / lambda')
    plt.tight_layout()
    # plt.savefig(result_save_dir + '/hamming_dist_same_loc.png')
    # plt.show()

    #%% scatter plot
    # get distance
    from genome import score
    CSI_data_all = np.concatenate(CSI_datas, axis=0)
    codeword_ref = best_genomes[0].predict(np.mean(CSI_data_ref, axis=0))
    dist_X, dist_C, _ = score(CSI_data_all, np.mean(CSI_data_ref, axis=0), codewords, codeword_ref, 1)

    plt.figure()
    plt.scatter(dist_X, dist_C)
    plt.xlabel('Euclidean distance')
    plt.ylabel('Hamming distance')
    # plt.savefig(result_save_dir + '/hamming_dist_scatter.png')
    #plt.show()

    # save codewords
    codewords = pd.DataFrame(data = np.concatenate(codewords, axis=0))
    codewords.to_csv(result_save_dir+'/codewords.csv', index=False)