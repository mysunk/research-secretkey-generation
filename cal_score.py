import pandas as pd
import numpy as np

### load dataset
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

# Channel State Information of bob (especially, gain)
X_bob = pd.read_csv('data/gain_1.csv', header=None)
X_bob = preprocessing(data = X_bob.values)
X_bob = np.mean(X_bob, axis=0)


# Channel State Information of bob or eve (especially, gain)
# train data
X = []
for i in range(1,7,1):
    CSI_data = pd.read_csv('data/gain_' + str(i) + '.csv', header=None)
    CSI_data = preprocessing(data = CSI_data.values)
    X.append(CSI_data)
X = np.concatenate(X, axis=0)


def fitness_score(X, X_bob, C, C_bob, CONST, POWER_RATIO):
    """
    :param X: CSI data of Bob or Alice
    :param X_bob: ground CSI data of Bob
    :param C: transformed codeword of X
    :param C_bob: transformed codeword of X_bob
    :param CONST: constant ratio factor in fitness function
    :param POWER_RATIO: power ratio factor in fitness function
    :return: 1) Euclidean distance of X, 2) hamming distance of C, 3) score value, 4) ground truth of fitness function
    """
    # vars
    cw_length = C.shape[1]
    X_length = X.shape[1]

    # calculate distance of X
    X_dist = np.linalg.norm(X_bob - X, axis=1)

    # calculate distance of C
    def hamming_distance(x, y):
        len_bit = max(len(x), len(y))
        bin_x = x
        bin_y = y

        result = 0
        for i in range(len_bit):
            if bin_x[i] != bin_y[i]:
                result += 1
        return result

    C_dist = np.zeros((C.shape[0]))
    for i in range(C.shape[0]):
        C_dist[i] = hamming_distance(np.ravel(C_bob), C[i])

    # calculate GNT for fitness function
    GNT = (X_dist / X_length) ** POWER_RATIO * CONST * cw_length

    # calculate score
    score_val = np.sqrt(np.mean(((X_dist / X_length) ** POWER_RATIO * CONST - C_dist / cw_length) ** 2))

    return X_dist, C_dist, score_val, GNT


def genome_score(genome, CONST=1, POWER_RATIO=1):
    """
    :param genome: genome object
    :param CONST: constant ratio factor in score function
    :param POWER_RATIO: power ratio factor in score function
    :return: genome object which contatin X_dist, C_dist and score value
    """
    C_GNT = genome.predict(X_bob)
    C = genome.predict(X)
    genome.X_dist, genome.C_dist, genome.score, _ = fitness_score(X, X_bob, C, C_GNT, CONST, POWER_RATIO)
    return genome