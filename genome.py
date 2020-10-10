import numpy as np
from scipy.special import expit
import scipy.special as sc
import pandas as pd
from util import *
from load_dataset import *

# load dataset
CSI_data = pd.read_csv('gain_1.csv', header=None)
CSI_data = CSI_data.values.T
CSI_data = minmax_norm(CSI_data)


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

    def softmax(self, x):
        return np.exp(x - sc.logsumexp(x))

    def linear(self, x):
        return x

    def forward(self, inputs):
        net = np.matmul(inputs, self.w1) + self.b1
        net = self.linear(net)
        net = np.matmul(net, self.w2) + self.b2
        net = self.linear(net)
        net = np.matmul(net, self.w3) + self.b3
        net = (net - np.mean(net)) / np.std(net)  # batch normalization
        net = self.linear(net)
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

def score(output):
    sample_num = output.shape[0]
    distances = 0
    for i in range(sample_num-1):
        distances += hamming_distance(output[i], output[i+1])
    distances_mean = distances / (sample_num-1)
    return distances_mean

def genome_score(genome):
    codewords = genome.predict(CSI_data)
    genome.score = score(codewords)
    return genome