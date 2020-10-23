import pandas as pd
import numpy as np
from genome import *
import matplotlib.pyplot as plt
import argparse
import matplotlib
font = {'size': 15, 'family': 'monospace'}
matplotlib.rc('font', **font)

# load all dataset
CSI_datas = []
num_samples = 27
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
    parser.add_argument("--result_save_dir", default='1021/1', type=str)
    parser.add_argument("--reference", default='1', type=str)
    parser.add_argument("--score_type", default='6', type=int)
    parser.add_argument("--CONST", default='1', type=float)
    parser.add_argument("--POWER_RATIO", default='0.5', type=float)

    args = parser.parse_args()
    result_save_dir = 'results/' + args.result_save_dir

    # load reference dataset
    CSI_data_ref = pd.read_csv('data_in_use/gain_' + args.reference + '.csv', header=None)
    CSI_data_ref = CSI_data_ref.values.T
    CSI_data_ref = minmax_norm(CSI_data_ref)

    import pickle
    with open(result_save_dir + '/best_genomes.pkl','rb') as f:
        best_genomes = pickle.load(f)

    #%% check hamming distance
    # predict
    codewords_ref = best_genomes[0].predict(CSI_data_ref)
    # plot
    hamm_dists = np.zeros((1000, len(CSI_datas)), dtype=int)
    dists_same_loc = np.zeros((1000, len(CSI_datas)), dtype=int)
    # random sampling
    max_i = CSI_data_ref.shape[0]
    codewords = []
    for j in range(len(CSI_datas)):
        codewords_2 = best_genomes[0].predict(CSI_datas[j])
        codewords.append(codewords_2)
        for i in range(1000):
            # for same sample
            hamm_dists[i, j] = hamming_distance(codewords_ref[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
            dists_same_loc[i,j] = hamming_distance(codewords_2[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
    codewords = np.concatenate(codewords, axis=0)

    ref_num = int(args.reference ) - 1
    locs = np.arange(-0.06*(ref_num), 0.06*(num_samples - ref_num), 0.06)
    locs = [float('{:.2f}'.format(l)) for l in locs]

    plt.figure(figsize=(10,5))
    plt.boxplot(x=dists_same_loc / 242)
    plt.xticks(range(num_samples)[::4], locs[::4])
    plt.ylabel('BER')
    plt.xlabel('d / lambda')
    plt.tight_layout()
    plt.savefig(result_save_dir + '/hamming_dist_same_loc.png')
    # plt.show()

    #%% scatter plot
    # get distance
    from genome import score
    CSI_data_all = np.concatenate(CSI_datas, axis=0)
    codeword_ref = best_genomes[0].predict(np.mean(CSI_data_ref, axis=0))
    dist_X, dist_C, _, score_GNT = score(CSI_data_all, np.mean(CSI_data_ref, axis=0), codewords, codeword_ref, args.score_type, args.CONST, args.POWER_RATIO)

    plt.figure()
    plt.scatter(dist_X, dist_C/242*100, label='result',facecolors='none',edgecolors='k')
    plt.xlabel('Euclidean distance')
    # plot ground truth for score function
    plt.plot(dist_X, score_GNT/242*100,'rx', label='GNT')
    plt.ylabel('BER [%]')
    plt.title(f'With reference {args.reference}')
    plt.legend()
    plt.savefig(result_save_dir + f'/hamming_dist_scatter_{args.reference}.png')
    # plt.show()

    score_GNT = np.array([np.mean(score_GNT[i*1000:(i+1)*1000], axis=0) for i in range(num_samples)])
    plt.figure(figsize=(10,5))
    plt.boxplot(x=hamm_dists)
    plt.plot(range(1, num_samples+1),score_GNT,'rx-', label='GNT')
    plt.xticks(range(1, num_samples+1)[::4], locs[::4])
    plt.ylabel('Hamming distance')
    plt.xlabel('d / lambda')
    plt.title(f'With reference {args.reference}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(result_save_dir + f'/hamming_dist_{args.reference}.png')
    # plt.show()


    # save codewords
    codewords = pd.DataFrame(data = np.concatenate(codewords, axis=0))
    codewords.to_csv(result_save_dir+'/codewords.csv', index=False)

    print('Process done...')