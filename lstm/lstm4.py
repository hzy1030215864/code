import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
xiaohao_data = pd.read_csv(r'E:\代码\lstm\data\BFG\热风炉消耗BFG.csv')

# 提取特征列
xiaohao = xiaohao_data["value"][:5000]
time = xiaohao_data["datetime"]

# 数据准备
scaler = MinMaxScaler()
xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 2 # 输入序列的长度
output_length = 1  # 输出序列的长度
X, y = [], []
for i in range(len(xiaohao) - sequence_length - output_length):
    X.append(xiaohao[i:i+sequence_length])
    y.append(xiaohao[i+sequence_length+output_length-1])  # 仅保留最后一个点作为输出

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8  # 可以调整此参数
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 模型创建
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=output_length))  # 输出层神经元数量为1，用于一个点的预测
model.compile(optimizer='adam', loss='mean_squared_error')

# 模型训练
model.fit(X_train, y_train, epochs=50, batch_size=50)

# 预测
y_pred = model.predict(X_test)
num_points = 50

# 反归一化预测值
y_pred_original = scaler.inverse_transform(y_pred)
y_true_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = y_pred_original[:num_points]
y_true_original = y_true_original[:num_points]

# 可视化比较
plt.plot(y_true_original, label="真实值")
plt.plot(y_pred_original, label="预测值")
plt.legend()
plt.show()
