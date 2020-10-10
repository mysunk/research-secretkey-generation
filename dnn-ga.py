'# -*- coding: utf-8 -*-'
"""
Created on Fri Jun  5 22:54:56 2020

@author: guseh
"""

import numpy as np
import multiprocessing
import warnings
from copy import deepcopy
from genome import Genome, genome_score
warnings.filterwarnings(action='ignore')
np.random.seed(76)

#%% Hyperparameters
CPU_CORE = multiprocessing.cpu_count()          # 멀티프로세싱 CPU 사용 수
N_POPULATION = 36                               # 세대당 생성수
N_BEST = 5                                      # 베스트 수
N_CHILDREN = 7                                  # 자손 유전자 수
PROB_MUTATION = 0.01                            # 돌연변이
mutation_std = 1                                # 돌연변이시 standard deviation
REVERSE = False                                 # 배열 순서 (False: ascending order, True: descending order) == maximize
score_ini = 10e+10                              # 초기 점수
batch_size = 32                                 # batch size
input_length = 242                              # subcarrier 수
output_length = 484                             # codeword length
h1 = 20                                         # 히든레이어1 노드 수
h2 = 26                                         # 히든레이어2 노드 수
h3 = 10                                         # 히든레이어3 노드 수
EPOCHS = 1000                                   # 반복 횟수
early_stopping = 30                             # saturation시 early stopping
crossover_fraction = 0.5                        # crossover 비율


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
while n_gen <= EPOCHS:
    genomes = np.array(genomes)    
    while len(genomes)%CPU_CORE != 0:
        genomes = np.append(genomes, Genome(score_ini, input_length, output_length, h1, h2, h3, batch_size,init_weight=None))
    genomes = genomes.reshape((len(genomes)//CPU_CORE, CPU_CORE))
    
    for idx, _genomes in enumerate(genomes):
        if __name__ == '__main__':
            pool = multiprocessing.Pool(processes=1)
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
    print('EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (n_gen, genomes[0].score, bs, s))    
    
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
        if np.argmax(sub_scores) == 0:
            print('No improvement, early stopping...')
            break

    n_gen += 1

print('==Process 2 Done==')

#%% plot result
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

plt.show()