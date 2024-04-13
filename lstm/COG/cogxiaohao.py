import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Concatenate, LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
import math
from keras.models import Sequential
from keras.layers import GRU, Dense
# 确保双向兼容
from keras.utils.layer_utils import count_params
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

xiaohao_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\COG\轧钢混合系统COG.csv')


xiaohao = xiaohao_data["value"][1:10000]

# 数据准备
scaler = MinMaxScaler()


xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 和你的 X 数据中时间步数一致
n_features = 1        # 你输入数据中的特征数量，这里以 4 个为例
n_output = 30
X, y = [], []
for i in range(len(xiaohao) - sequence_length - n_output):
    X.append(np.c_[xiaohao[i:i+sequence_length]])
    y.append(xiaohao[i+sequence_length:i+sequence_length+n_output])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# 定义模型
model = Sequential()
model.add(LSTM(units=128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=128, activation='tanh', return_sequences=True))
model.add(LSTM(units=128, activation='tanh'))
model.add(Dense(units=n_output))  # 输出层神经元数量为10，用于预测10个点
model.compile(optimizer='adam', loss='mse')

# 编译模型

# 注意 X 和 y 的形状应该匹配模型的输入和输出要求
# 训练模型
history = model.fit(X, y, epochs=100, batch_size=50, validation_split=0.15)

y_pred = model.predict(X_test)

 #反归一化预测值和真实值
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler.inverse_transform(y_test.reshape(-1, 1))[:n_output]

plt.plot(range(n_output), y_true_original, label="真实值", marker='o')
plt.plot(range(n_output), y_pred_original, label="预测值", marker='x')
# plt.title('LSTMCOG消耗量预测')
plt.legend()
# 重塑 y_test 和 y_pred，使它们维度相同
y_test_reshaped = y_test.reshape(-1, n_output)
y_pred_reshaped = y_pred.reshape(-1, n_output)

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test_reshaped, y_pred_reshaped)
rmse = math.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
print('MSE:', mse)
print('RMSE:', rmse)
print(mape(y_true_original, y_pred_original)) # 76.07142857142858，即76%
plt.show()
