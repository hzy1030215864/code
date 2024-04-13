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
# 加载数据
bfgfasheng_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\BFG\BFGFSL.csv')
fengya_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\BFG\FengYa.csv')
fl_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\BFG\FL.csv')
fy_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\BFG\FY.csv')
xiaohao_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\BFG\热风炉消耗BFG.csv')

# 提取特征列
bfgfasheng = bfgfasheng_data["value"][1:5000]
fengya = fengya_data["value"][1:5000]
fl = fl_data["value"][1:5000]
fy = fy_data["value"][1:5000]
xiaohao = xiaohao_data["value"][1:5000]
time = xiaohao_data["datetime"]

# 数据准备
scaler = MinMaxScaler()
fengya = scaler.fit_transform(fengya.values.reshape(-1, 1))
fl = scaler.fit_transform(fl.values.reshape(-1, 1))
fy = scaler.fit_transform(fy.values.reshape(-1, 1))

xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))
scaler_bfgfasheng = MinMaxScaler() # 专门用于bfgfasheng的反归一化
bfgfasheng = scaler_bfgfasheng.fit_transform(bfgfasheng.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 和你的 X 数据中时间步数一致
n_features = 4        # 你输入数据中的特征数量，这里以 4 个为例
n_output = 30
X, y = [], []
for i in range(len(xiaohao) - sequence_length - n_output):
    X.append(np.c_[xiaohao[i:i+sequence_length], fengya[i:i+sequence_length], fl[i:i+sequence_length], fy[i:i+sequence_length]])
    y.append(bfgfasheng[i+sequence_length:i+sequence_length+n_output])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# 定义模型
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=False, input_shape=(sequence_length, n_features)))
model.add(Dense(n_output))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 注意 X 和 y 的形状应该匹配模型的输入和输出要求
# 训练模型
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.15)

y_pred = model.predict(X_test)

 #反归一化预测值和真实值
y_pred_original = scaler_bfgfasheng.inverse_transform(y_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:n_output]

# 可视化比较
plt.plot(range(n_output), y_true_original, label="真实值")
plt.plot(range(n_output), y_pred_original, label="预测值")
plt.title('LSTM')

plt.legend()
# 重塑 y_test 和 y_pred，使它们维度相同
y_test_reshaped = y_test.reshape(-1, n_output)
y_pred_reshaped = y_pred.reshape(-1, n_output)

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test_reshaped, y_pred_reshaped)
rmse = math.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
print('MSE:', mse)
print('RMSE:', rmse)
print(mape(y_test_reshaped, y_pred_reshaped)) # 76.07142857142858，即76%
plt.show()
