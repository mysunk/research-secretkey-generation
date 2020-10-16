from scipy.special import expit
import numpy as np
import pandas as pd

def minmax_norm(CSI_data1):
    max_v1 = np.max(CSI_data1)
    min_v1 = np.min(CSI_data1)
    CSI_data1 = (CSI_data1 - min_v1) / (max_v1 - min_v1)
    return (CSI_data1)

# load GNT
X_GNT = pd.read_csv('data_in_use/gain_1.csv', header=None)
X_GNT = X_GNT.values.T
X_GNT = minmax_norm(X_GNT)
X_GNT = np.mean(X_GNT, axis=0)

# load all data
CSI_datas = []
for i in range(1,9,2):
    CSI_data = pd.read_csv('data_in_use/gain_' + str(i) + '.csv', header=None)
    # Transpose
    CSI_data = CSI_data.values.T
    # Min-max normalization
    CSI_data = minmax_norm(CSI_data)
    CSI_datas.append(CSI_data)
CSI_datas = np.concatenate(CSI_datas, axis=0)

class network():

    def __init__(self, input_len, output_len, hidden_layer1, hidden_layer2, hidden_layer3, init_weight):
        # initialize weight
        self.w1 = np.random.randn(input_len, hidden_layer1) * np.sqrt(2 / (input_len + hidden_layer1))
        self.w2 = np.random.randn(hidden_layer1, hidden_layer2) * np.sqrt(2 / (hidden_layer1 + hidden_layer2))
        self.w3 = np.random.randn(hidden_layer2, hidden_layer3) * np.sqrt(2 / (hidden_layer2 + hidden_layer3))
        self.w4 = np.random.randn(hidden_layer3, output_len) * np.sqrt(2 / (hidden_layer3 + output_len))
        self.b1 = np.random.randn(1, hidden_layer1) * np.sqrt(1 / hidden_layer1)
        self.b2 = np.random.randn(1, hidden_layer2) * np.sqrt(1 / hidden_layer2)
        self.b3 = np.random.randn(1, hidden_layer3) * np.sqrt(1 / hidden_layer3)
        self.b4 = np.random.randn(1, output_len) * np.sqrt(1 / output_len)


    def sigmoid(self, x):
        return expit(x)

    def relu(self, x):
        return x * (x > 0)

    def linear(self, x):
        return x

    def forward(self, inputs):
        net = np.matmul(inputs, self.w1) + self.b1
        net = self.relu(net)
        net = np.matmul(net, self.w2) + self.b2
        net = self.relu(net)
        net = np.matmul(net, self.w3) + self.b3
        net = (net - np.mean(net)) / np.std(net)  # batch normalization
        net = self.relu(net)
        net = np.matmul(net, self.w4) + self.b4
        net = (net - np.mean(net)) / np.std(net)  # batch normalization
        score = self.sigmoid(net)
        # 이부분 수정 필요
        score[score>0.5] = 1
        score = score.astype(int)
        return score


class Genome():

    def __init__(self, score_ini, input_length, output_length, h1, h2, h3, batch_size, init_weight=None):
        self.network = network(input_length, output_length, h1, h2, h3, init_weight=None)

        # 평가 점수 초기화
        self.score = score_ini

        self.X_dist = None
        self.C_dist = None

    def predict(self, data):
        codewords = self.network.forward(data)
        return codewords

def hamming_distance(x, y):
    len_bit = max(len(x), len(y))
    bin_x = x
    bin_y = y

    result = 0
    for i in range(len_bit):
        if bin_x[i] != bin_y[i]:
            result += 1
    return result

def score(X, X_GNT, C, C_GNT, score_type):
    # vars
    cw_length = C.shape[1]
    X_length = X.shape[1]

    # calculate distance of X
    X_dist = np.linalg.norm(X_GNT - X, axis=1)

    # calculate distance of C
    C_dist = np.zeros((C.shape[0]))
    for i in range(C.shape[0]):
        C_dist[i] = hamming_distance(np.ravel(C_GNT), C[i])

    # score function
    CONST = 1
    if score_type == 1:
        ratio_factor = CONST * cw_length / X_dist
        return X_dist, C_dist, np.sqrt(np.mean((X_dist / X_length * ratio_factor - C_dist / cw_length) ** 2))
    elif score_type == 2:
        ratio_factor = CONST * cw_length / (X_dist ** 2)
        return X_dist, C_dist, np.sqrt(np.mean((X_dist / X_length * ratio_factor - C_dist / cw_length) ** 2))
    elif score_type == 3:
        return X_dist, C_dist, np.sqrt(np.mean((X_dist**2 / X_length - C_dist / cw_length) ** 2))


def genome_score(genome, score_type):
    codeword_ref = genome.predict(X_GNT)
    codewords = genome.predict(CSI_datas)
    genome.X_dist, genome.C_dist, genome.score = score(CSI_datas,X_GNT, codewords, codeword_ref, score_type)
    return genome