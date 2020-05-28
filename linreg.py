import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calc_accuracy(label, pred):
    SS_res = 0
    SS_tot = 0
    avg = np.sum(label)/len(label)
    for i in range(len(label)):
        SS_res += (label[i] - pred[i])**2
        SS_tot += (label[i] - avg)**2

    R_2 = 1 - SS_res/SS_tot
    return R_2

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# choose training accuracy
x_test = 1-test_set[['train_accs_49']].to_numpy()

# choose validation accuracy
x_test_2 = 1-test_set[['val_accs_49']].to_numpy()

# choose train error
y = train_set[['train_error']].to_numpy()
x = 1 - train_set[['train_accs_49']].to_numpy()

y_2 = train_set[['val_error']].to_numpy()
x_2 = 1 - train_set[['val_accs_49']].to_numpy()

regressor = LinearRegression()

# linear regression for train set
regressor.fit(x, y)
pred_y = regressor.predict(x)
pred_y_test = regressor.predict(x_test)
accu_1 = calc_accuracy(y,pred_y)
print(accu_1)

# linear regression for validation set
regressor.fit(x_2,y_2)
pred_y_2 = regressor.predict(x_2)
pred_y_test_2 = regressor.predict(x_test_2)
accu_2 = calc_accuracy(y_2,pred_y_2)
print(accu_2)

plt.subplot(2,1,1)
plt.plot(x,pred_y,c='red')
plt.scatter(x, y)
plt.xlabel("final train error")
plt.ylabel("train error on epoch 50")

plt.subplot(2,1,2)
plt.plot(x_2,pred_y_2,c='red')
plt.scatter(x_2, y_2)
plt.xlabel("final validation error")
plt.ylabel("validation error on epoch 50")

plt.show()

result = []

for i in range(len(pred_y_test_2)):
    result.append(pred_y_test_2[i])
    result.append(pred_y_test[i])

result = np.asarray(result)

for i in range(len(result)):
    if result[i]<0:
        result[i] = 0


df = pd.DataFrame(data=result)
df.to_csv('test_results_linreg.csv', encoding='utf-8', index=False)