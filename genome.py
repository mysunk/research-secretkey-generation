from scipy.special import expit
from AE.load_dataset import *

# load GNT
CSI_data_ref = pd.read_csv('data_in_use/gain_1.csv', header=None)
CSI_data_ref = CSI_data_ref.values.T
CSI_data_ref = minmax_norm(CSI_data_ref)
X_GNT = np.mean(CSI_data_ref, axis=0)

# load all
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

def score(CSI_data, output, codeword_ref):
    # sample_num = output.shape[0]
    # distances = 0
    # for i in range(sample_num-1):
    #     distances += hamming_distance(output[i], output[i+1])
    # distances_mean = distances / (sample_num-1)

    distance_X = np.linalg.norm(X_GNT - CSI_data)
    distances_C = np.zeros((output.shape[0]))

    for i in range(output.shape[0]):
        distances_C[i] = hamming_distance(codeword_ref, output[i])

    return np.abs(distance_X.mean() - distances_C.mean()) ** 2

def genome_score(genome):
    codeword_ref = genome.predict(X_GNT)
    codewords = genome.predict(CSI_datas)
    genome.score = score(CSI_datas, codewords, codeword_ref)
    return genome