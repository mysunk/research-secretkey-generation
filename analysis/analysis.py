#%% result
train_result = list_train[0].cpu().detach().numpy()
test_result = list_test[0].cpu().detach().numpy()
plt.plot(train_result[0],'-.')
plt.plot(train_result[1],'-x')
plt.show()

#%% Accuracy
import seaborn as sns
acc = (test_result[0] == test_result).mean(axis=1)
sns.boxplot(acc)
plt.show()