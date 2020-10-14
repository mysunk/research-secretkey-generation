from scipy.special import expit
import pandas as pd
import numpy as np
def minmax_norm(CSI_data1):
    max_v1 = np.max(CSI_data1)
    min_v1 = np.min(CSI_data1)
    CSI_data1 = (CSI_data1 - min_v1) / (max_v1 - min_v1)
    return (CSI_data1)

# load GNT
CSI_data_ref = pd.read_csv('data_in_use/gain_1.csv', header=None)
CSI_data_ref = CSI_data_ref.values.T
CSI_data_ref = minmax_norm(CSI_data_ref)
X_GNT = np.mean(CSI_data_ref, axis=0)

# load all
CSI_datas = []
for i in range(1,11,1):
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
        score[score > 0.5] = 1
        score = score.astype(int)
        return score


class Genome():

    def __init__(self, score_ini, input_length, output_length, h1, h2, h3, batch_size, init_weight=None):
        self.network = network(input_length, output_length, h1, h2, h3, init_weight=None)

        self.dist_X = None
        self.dist_C = None

        # 평가 점수 초기화
        self.score = score_ini

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

def score(X, output, C_GNT, score_type):
    # calculate distance of X and C
    distance_X = np.linalg.norm(X_GNT - X, axis=1)
    distances_C = np.zeros((output.shape[0]))
    for i in range(output.shape[0]):
        distances_C[i] = hamming_distance(np.ravel(C_GNT), output[i])
    # vaying score function
    if score_type == 1:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = 0.5 / distance_X[distance_X > 1]
    elif score_type == 2:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.exp(distance_X[distance_X > 1])
    else:
        raise NotImplementedError('Not implemented type of score function')
    score_val = np.mean(np.abs(distance_X * ratio_factor - distances_C / output.shape[1]))
    return distance_X, distances_C, score_val


def genome_score(genome, score_type):
    codeword_ref = genome.predict(X_GNT)
    codewords = genome.predict(CSI_datas)
    genome.dist_X, genome.dist_C, genome.score = score(CSI_datas, codewords, codeword_ref, score_type)
    return genome

""" 1013
 # varying ratio factor
    if score_type == 1:
        ratio_factor = 1
    elif score_type == 2:
        ratio_factor = 30
    elif score_type == 3:
        ratio_factor = np.exp(distance_X)
    elif score_type == 4:
        ratio_factor = np.exp(distance_X)*3
    elif score_type == 5:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.random.randn((distance_X > 1).sum()) * np.sqrt(0.1) + 17.5
    elif score_type == 6:
        ratio_factor = np.nanmax(distances_C / distance_X)
    elif score_type == 7:
        ratio_factor = np.median(codeword_ref) / np.median(X_GNT)
    elif score_type == 8:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.random.randn((distance_X > 1).sum()) * np.sqrt(0.1) + 30
    elif score_type == 9:
        ratio_factor = np.exp(distance_X)*5
    elif score_type == 10:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = 242 / distance_X[distance_X > 1]
    elif score_type == 11:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.random.randn((distance_X > 1).sum()) * np.sqrt(0.1) + 40
    elif score_type == 12:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.random.randn((distance_X > 1).sum()) * np.sqrt(0.1) + 50
    elif score_type == 13:
        ratio_factor = np.zeros(distance_X.shape)
        ratio_factor[distance_X <= 1] = 0
        ratio_factor[distance_X > 1] = np.random.randn((distance_X > 1).sum()) * np.sqrt(0.1) + 60
    else:
        raise NotImplementedError('Not implemented type of score function')
"""