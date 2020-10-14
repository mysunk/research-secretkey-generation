import pandas as pd
data = pd.read_csv('distances.csv')
X_dist = data.iloc[:,1]
C_dist = data.iloc[:,2]

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.distplot(X_dist)
# plt.show()
plt.figure()
sns.distplot(C_dist)
# plt.show()