import pandas as pd
from AE.load_dataset import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 15}
matplotlib.rc('font', **font)

CSI_datas = []
for i in range(1, 42):
    CSI_data = pd.read_csv('data_in_use/gain_' + str(i) + '.csv', header=None)
    # Transpose
    CSI_data = CSI_data.values.T
    # Min-max normalization
    CSI_data = minmax_norm(CSI_data)
    CSI_datas.append(CSI_data)

#%% plot dataset
plt.figure()
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(CSI_datas[i-1].T)
    plt.title(i)
    # plt.xlabel('Subcarrier index')
plt.show()

#%% calculate correlation and distance
corrs = np.zeros((1000, len(CSI_datas)))
distances = np.zeros((1000, len(CSI_datas)))
max_i = 1000
for j in range(len(CSI_datas)):
    for i in range(1000):
        corrs[i, j] = np.corrcoef(CSI_datas[0][np.random.randint(0, max_i, 1)[0], :], CSI_datas[j][np.random.randint(0, max_i, 1)[0], :])[0,1]
        distances[i, j] = np.linalg.norm(CSI_datas[0][np.random.randint(0, max_i, 1)[0], :]-CSI_datas[j][np.random.randint(0, max_i, 1)[0], :])

#%% plot correlation and distance
locs = np.arange(0, 0.06 * 41, 0.06)
plt.figure(figsize=(6,4))
plt.subplot(2,1,1)
plt.boxplot(x = corrs[:,:41])
plt.xticks(range(41)[::4], locs[::4])
plt.xlabel('d/lambda')
plt.ylabel('Correlation')
plt.subplot(2,1,2)
plt.boxplot(x = distances[:,:41])
plt.xticks(range(41)[::4], locs[::4])
plt.xlabel('d/lambda')
plt.ylabel('Distance')
plt.show()

#%% Distribution of distance
import seaborn as sns

for i in range(0,40,1):
    sns.distplot(distances[:,i], label=i, hist=True)
plt.xlabel('Distance')
plt.ylabel('Count')
plt.title('Distribution of euclidean distance')
# plt.legend()
plt.show()

for i in range(0,40,1):
    sns.distplot(corrs[:,i], label=i, hist=True)
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.title('Distribution of correlation')
# plt.legend()
plt.show()

#%% plot mean
plt.plot(CSI_datas[0].mean(axis=0))
plt.ylabel('Gain')
plt.xlabel('Subcarrier index')
plt.title('GNT')
plt.show()

#%%
codewords = pd.read_csv('results/1011/5/codewords.csv')
codewords_sub = codewords.iloc[np.arange(0,14000,1000),:]
import seaborn as sns
sns.heatmap(codewords_sub.T)
plt.show()

#%%
plt.plot(codewords.iloc[0,:].values)
plt.show()