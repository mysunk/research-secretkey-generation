import pandas as pd
import numpy as np
from cal_score import fitness_score
import matplotlib.pyplot as plt
import argparse
import matplotlib
import pickle
font = {'size': 15, 'family': 'Times New Roman'}
matplotlib.rc('font', **font)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_save_dir", default='results/', type=str)
    parser.add_argument("--reference", default='1', type=str)
    parser.add_argument("--CONST", default='0.8', type=float)
    parser.add_argument("--POWER_RATIO", default='0.5', type=float)

    args = parser.parse_args()
    result_save_dir = args.result_save_dir

    """
    1. Load dataset
    """
    def preprocessing(data):
        """
        :param data: Input data
        :return: transpose and min-max normalized data
        """
        # Transpose
        data = data.T

        # Min-max normalization
        def minmax_norm(data):
            max_ = np.max(data)
            min_ = np.min(data)
            data = (data - min_) / (max_ - min_)
            return data

        data = minmax_norm(data)
        return data

    CSI_datas = []
    num_samples = 27
    for i in range(1, num_samples + 1):
        CSI_data = pd.read_csv('data/gain_' + str(i) + '.csv', header=None)
        CSI_data = preprocessing(CSI_data.values)
        CSI_datas.append(CSI_data)

    # load reference dataset (i.e., bob)
    CSI_data_ref = pd.read_csv('data/gain_' + args.reference + '.csv', header=None)
    CSI_data_ref = preprocessing(CSI_data_ref.values)

    # load weight of dnn
    with open(result_save_dir + '/best_genomes.pkl','rb') as f:
        best_genomes = pickle.load(f)

    """
    2. Check hamming distance
    """
    # predict
    codewords_ref = best_genomes[0].predict(CSI_data_ref)

    # plot
    hamm_dists = np.zeros((1000, len(CSI_datas)), dtype=int)
    dists_same_loc = np.zeros((1000, len(CSI_datas)), dtype=int)

    # calcluate hamming distance
    max_i = CSI_data_ref.shape[0]
    codewords = []
    def hamming_distance(x, y):
        len_bit = max(len(x), len(y))
        bin_x = x
        bin_y = y

        result = 0
        for i in range(len_bit):
            if bin_x[i] != bin_y[i]:
                result += 1
        return result
    for j in range(len(CSI_datas)):
        codewords_2 = best_genomes[0].predict(CSI_datas[j])
        codewords.append(codewords_2)
        # random sampling for 1000 times
        for i in range(1000):
            hamm_dists[i, j] = hamming_distance(codewords_ref[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
            dists_same_loc[i,j] = hamming_distance(codewords_2[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
    codewords = np.concatenate(codewords, axis=0)


    """
    3. Create plots about hamming distance of codewords w.r.t euclidean distance of CSIs
    """

    # get distance
    CSI_data_all = np.concatenate(CSI_datas, axis=0)
    codeword_ref = best_genomes[0].predict(np.mean(CSI_data_ref, axis=0))
    dist_X, dist_C, _, score_GNT = fitness_score(CSI_data_all, np.mean(CSI_data_ref, axis=0), codewords, codeword_ref, args.CONST, args.POWER_RATIO)

    plt.figure()
    plt.scatter(dist_X, dist_C/242*100, label='Proposed',facecolors='none',edgecolors='k')
    plt.xlabel('Euclidean distance')
    # plot ground truth for score function
    plt.plot(dist_X, score_GNT/242*100,'rx', label='Ideal')
    plt.ylabel('PHD')
    plt.title(f'Percential hamming distance w.r.t euclidean distance')
    plt.legend()
    plt.savefig(result_save_dir + f'/hamming_dist_scatter_{args.reference}.png')
    plt.show()

    score_GNT = np.array([np.mean(score_GNT[i*1000:(i+1)*1000], axis=0) for i in range(num_samples)])
    plt.figure(figsize=(10,5))
    plt.boxplot(x=hamm_dists)
    plt.plot(range(1, num_samples+1),score_GNT,'rx-', label='GNT')
    ref_num = int(args.reference) - 1
    locs = np.arange(-0.06 * (ref_num), 0.06 * (num_samples - ref_num), 0.06)
    locs = [float('{:.2f}'.format(l)) for l in locs]
    plt.xticks(range(1, num_samples+1)[::4], locs[::4])
    plt.ylabel('Hamming distance')
    plt.xlabel('d / lambda')
    plt.title(f'Hamming distance w.r.t d / lambda')
    plt.tight_layout()
    plt.legend()
    plt.savefig(result_save_dir + f'/hamming_dist_{args.reference}.png')
    plt.show()

    # save codewords
    codewords = pd.DataFrame(data = np.concatenate(codewords, axis=0))
    codewords.to_csv(result_save_dir+'/codewords.csv', index=False)