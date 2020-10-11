'# -*- coding: utf-8 -*-'
"""
Created on Fri Jun  5 22:54:56 2020

@author: guseh
"""

import multiprocessing
import warnings
from copy import deepcopy
from genome import Genome, genome_score, hamming_distance
import time
import argparse
from AE.load_dataset import *
warnings.filterwarnings(action='ignore')
np.random.seed(76)

# User input
parser = argparse.ArgumentParser()
parser.add_argument("--N_POPULATION", default=36, type=int)
parser.add_argument("--N_BEST", default=5, type=int)
parser.add_argument("--N_CHILDREN", default=7, type=int)
parser.add_argument("--PROB_MUTATION", default=0.01, type=float)
parser.add_argument("--mutation_std", default=1, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--input_length", default=242, type=int)
parser.add_argument("--output_length", default=484, type=int)
parser.add_argument("--EPOCHS", default=3, type=int)
parser.add_argument("--early_stopping", default=30, type=int)
parser.add_argument("--h1", default=128, type=int)
parser.add_argument("--h2", default=128, type=int)
parser.add_argument("--h3", default=128, type=int)
parser.add_argument("--result_save_dir", default='tmp', type=str)
parser.add_argument("--crossover_fraction", default=0.5, type=float)
args = parser.parse_args()


#%% Hyperparameters
CPU_CORE = multiprocessing.cpu_count()          # 멀티프로세싱 CPU 사용 수
REVERSE = False                                 # 배열 순서 (False: ascending order, True: descending order) == maximize
score_ini = 10e+10                              # 초기 점수
batch_size = args.batch_size                    # batch size
input_length = args.input_length                # subcarrier 수
output_length = args.output_length              # codeword length
h1 = args.h1                                    # 히든레이어1 노드 수
h2 = args.h2                                    # 히든레이어2 노드 수
h3 = args.h3                                    # 히든레이어3 노드 수
EPOCHS = args.EPOCHS                            # 반복 횟수
early_stopping = args.early_stopping            # saturation시 early stopping
crossover_fraction = args.crossover_fraction    # crossover 비율
N_POPULATION = args.N_POPULATION                # 세대당 생성수
N_BEST = args.N_BEST                            # 베스트 수
N_CHILDREN = args.N_CHILDREN                    # 자손 유전자 수
PROB_MUTATION = args.PROB_MUTATION              # 돌연변이
mutation_std = args.mutation_std                # 돌연변이시 standard deviation

# plot save dir
result_save_dir = 'results/'+args.result_save_dir
import os
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

#%% Initial guess
genomes = []
for _ in range(N_POPULATION):
    genome = Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size, init_weight=None)
    genomes.append(genome)
try:
    for i in range(N_BEST):
        genomes[i] = best_genomes[i]
except:
    best_genomes = []
    for _ in range(N_BEST): # genome의 개수 조정
        genome = Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size, init_weight=None)
        best_genomes.append(genome)
print('==Process 1 Done==')

def crossover(process_1, process_2, new_process, weight, crossover_fraction):
    # crossover
    new_weight = np.zeros(getattr(new_process, weight).shape)
    for j in range(getattr(new_process, weight).shape[0]):
        w_e = getattr(new_process, weight).shape[1]
        cut = np.zeros(w_e)
        cut[np.random.choice(range(w_e), (int)(np.floor(w_e * crossover_fraction)))] = 1
        new_weight[j, cut == 1] = getattr(process_1, weight)[j, cut == 1]
        new_weight[j, cut == 0] = getattr(process_2, weight)[j, cut == 0]
    return new_weight

def mutation(new_process, mean, stddev, weight):
    # mutation event
    w_e = getattr(new_process, weight).shape[0]
    h_e = getattr(new_process, weight).shape[1]
    if np.random.uniform(0, 1) < PROB_MUTATION:
        new_weight = new_process.__dict__[weight] * np.random.normal(mean, stddev, size=(w_e, h_e)) * np.random.randint(0, 2, (w_e, h_e))
    else:
        new_weight = new_process.__dict__[weight]
    return new_weight

#%% 모델 학습
n_gen = 1 
score_history = []
high_score_history = []
mean_score_history = []
t_start = time.time()
while n_gen <= EPOCHS:
    genomes = np.array(genomes)    
    while len(genomes)%CPU_CORE != 0:
        genomes = np.append(genomes, Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size,init_weight=None))
    genomes = genomes.reshape((len(genomes)//CPU_CORE, CPU_CORE))
    
    for idx, _genomes in enumerate(genomes):
        if __name__ == '__main__':
            # hyper parameters
            pool = multiprocessing.Pool(processes=CPU_CORE)
            genomes[idx] = pool.map(genome_score, _genomes)
            pool.close()
            pool.join()
    genomes = list(genomes.reshape(genomes.shape[0]*genomes.shape[1]))    
    
     # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)
    
    # 평균 점수
    s = 0 
    for i in range(N_BEST):
        s += genomes[i].score
    s /= N_BEST
    
    # Best Score
    bs = genomes[0].score 
    
    # Best Model 추가
    if best_genomes is not None:
        genomes.extend(best_genomes)
        
    # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)
    
    score_history.append([n_gen, genomes[0].score])
    high_score_history.append([n_gen, bs])
    mean_score_history.append([n_gen, s])
    
    # 결과 출력
    t = time.time()
    epoch_duration = (t - t_start)/n_gen
    print('EPOCH #{}\tHistory Best Score: {:.2f}\tBest Score: {:.2f}\tMean Score: {:.2f}\t Avg time for 1 epoch: {:.2f} seconds'.format(n_gen, genomes[0].score, bs, s, epoch_duration))
    
    # 모델 업데이트
    best_genomes = deepcopy(genomes[:N_BEST])

    # CHILDREN 생성 -- 부모로부터 crossover을 해서 children 생성
    for i in range(N_CHILDREN):
        new_genome = deepcopy(best_genomes[0])
        genome_1 = np.random.choice(best_genomes)
        genome_2 = np.random.choice(best_genomes)
        for weight in ['w1','w2','w3','w4','b1','b2','b3','b4']:
            new_genome.network.__dict__[weight] = crossover(genome_1.network, genome_2.network, new_genome.network, weight, crossover_fraction)
        genomes.append(new_genome)

    # 모델 초기화
    genomes = []
    for i in range(int(N_POPULATION / len(best_genomes))):
        for bg in best_genomes:
            new_genome = deepcopy(bg)
            mean = 0
            stddev = mutation_std
            # Mutation (PROB_MUTATION의 확률로)
            for weight in ['w1','w2','w3','w4','b1','b2','b3','b4']:
                new_genome.network.__dict__[weight] = mutation(new_genome.network, mean, stddev, weight)
            genomes.append(new_genome)
            
    if REVERSE:
        if bs < score_ini:
            genomes[len(genomes)//2:] = [Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size,init_weight=None) for _ in range(N_POPULATION//2)]
    else:
        if bs > score_ini:
            genomes[len(genomes)//2:] = [Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size,init_weight=None) for _ in range(N_POPULATION//2)]

    # early stopping
    if n_gen > early_stopping:
        last_scores = high_score_history[-1 * early_stopping:]
        sub_scores = list(map(lambda x: x[1], last_scores))
        if np.argmin(sub_scores) == 0:
            print('No improvement, early stopping...')
            EPOCHS = n_gen
            break

    n_gen += 1

print('==Process 2 Done==')

#%% plot triain result
import matplotlib
font = {'size':15}
matplotlib.rc('font',**font)

import matplotlib.pyplot as plt
score_history = np.array(score_history)
high_score_history = np.array(high_score_history)
mean_score_history = np.array(mean_score_history)

plt.plot(score_history[:,0], score_history[:,1], '-o', label='BEST')
plt.plot(high_score_history[:,0], high_score_history[:,1], '-o', label='High')
plt.plot(mean_score_history[:,0], mean_score_history[:,1], '-o', label='Mean')
plt.legend()
plt.xlim(0, EPOCHS)
plt.ylim(bottom=0)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.savefig(result_save_dir+'/ga_train.png')
plt.show()

#%% check hamming distance
# load dataset
CSI_data_ref = pd.read_csv('data_in_use/gain_1.csv', header=None)
CSI_data_ref = CSI_data_ref.values.T
CSI_data_ref = minmax_norm(CSI_data_ref)

# load all
CSI_datas = []
for i in range(1,8):
    CSI_data = pd.read_csv('data_in_use/gain_' + str(i) + '.csv', header=None)
    # Transpose
    CSI_data = CSI_data.values.T
    # Min-max normalization
    CSI_data = minmax_norm(CSI_data)
    CSI_datas.append(CSI_data)

# predict
codewords_ref = best_genomes[0].predict(CSI_data_ref)

# plot
dists = np.zeros((100, len(CSI_datas)), dtype=int)
dists_same_loc = np.zeros((100, len(CSI_datas)), dtype=int)
# random sampling
max_i = CSI_data_ref.shape[0]
codewords = []
for j in range(len(CSI_datas)):
    codewords_2 = best_genomes[0].predict(CSI_datas[j])
    codewords.append(codewords_2)
    for i in range(100):
        # for same sample
        dists[i, j] = hamming_distance(codewords_ref[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])
        dists_same_loc[i,j] = hamming_distance(codewords_2[np.random.randint(0, max_i, 1)[0],:], codewords_2[np.random.randint(0, max_i, 1)[0],:])

locs = np.arange(0, 0.06*7, 0.06)

plt.figure(figsize=(10,5))
plt.boxplot(x=dists)
plt.xticks(range(7)[::2], locs[::2])
plt.ylabel('Hamming distance')
# plt.xlabel('Sample location #')
plt.xlabel('d / lambda')
plt.savefig(result_save_dir + '/hamming_dist.png')
# plt.show()

plt.figure(figsize=(10,5))
plt.boxplot(x=dists_same_loc)
plt.xticks(range(7)[::2], locs[::2])
plt.ylabel('Hamming distance')
# plt.xlabel('Sample location #')
plt.xlabel('d / lambda')
plt.savefig(result_save_dir + '/hamming_dist_same_loc.png')
# plt.show()

# save codewords
codewords = pd.DataFrame(data = np.concatenate(codewords, axis=0))
codewords.to_csv(result_save_dir+'/codewords.csv', index=False)