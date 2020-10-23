from scipy.special import expit
import numpy as np


class network():

    def __init__(self, input_len, output_len, hidden_layer1, hidden_layer2, hidden_layer3, init_weight = None):
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
        # convert from float to binary value
        score[score>0.5] = 1
        score = score.astype(int)
        return score


class Genome():

    def __init__(self, score_ini, input_length, output_length, h1, h2, h3, batch_size = None, init_weight=None):

        self.network = network(input_length, output_length, h1, h2, h3)
        self.score = score_ini
        self.X_dist = None
        self.C_dist = None

    def predict(self, data):
        codewords = self.network.forward(data)
        return codewords
