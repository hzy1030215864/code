import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from sklearn.metrics import mean_squared_error
import math

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
bfgfasheng = bfgfasheng_data["value"][1:10000]
fengya = fengya_data["value"][1:10000]
fl = fl_data["value"][1:10000]
fy = fy_data["value"][1:10000]
xiaohao = xiaohao_data["value"][1:10000]
time = xiaohao_data["datetime"]

# 数据准备
scaler = MinMaxScaler()
fengya = scaler.fit_transform(fengya.values.reshape(-1, 1))
fl = scaler.fit_transform(fl.values.reshape(-1, 1))
fy = scaler.fit_transform(fy.values.reshape(-1, 1))

bfgfasheng = scaler.fit_transform(bfgfasheng.values.reshape(-1, 1))
scaler_bfgfasheng = MinMaxScaler() # 专门用于bfgfasheng的反归一化
xiaohao = scaler_bfgfasheng.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 和你的 X 数据中时间步数一致
n_features = 4        # 你输入数据中的特征数量，这里以 4 个为例
n_output = 30
X, y = [], []
for i in range(len(xiaohao) - sequence_length - n_output):
    X.append(np.c_[bfgfasheng[i:i+sequence_length], fengya[i:i+sequence_length], fl[i:i+sequence_length], fy[i:i+sequence_length]])
    y.append(xiaohao[i+sequence_length:i+sequence_length+n_output])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 定义带有CNN的LSTM模型
model_cnn_lstm = Sequential()
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
model_cnn_lstm.add(LSTM(units=128, activation='tanh', return_sequences=True))
model_cnn_lstm.add(LSTM(units=128, activation='tanh', return_sequences=True))
model_cnn_lstm.add(LSTM(units=128, activation='tanh'))
model_cnn_lstm.add(Dense(units=n_output))  # 输出层神经元数量为10，用于预测10个点
model_cnn_lstm.compile(optimizer='adam', loss='mse')

# 训练带有CNN的LSTM模型
history_cnn_lstm = model_cnn_lstm.fit(X_train, y_train, epochs=50, batch_size=50, validation_split=0.15)

# 绘制预测曲线

y_pred_cnn_lstm = model_cnn_lstm.predict(X_test)
y_pred_cnn_lstm_original = scaler_bfgfasheng.inverse_transform(y_pred_cnn_lstm.reshape(-1, 1))[:n_output]

# 从Excel表中读取数据
excel_data = pd.read_excel('E:\新建文件夹\代码\lstm\BFG\最后30个点预测结果.xlsx')

# 绘制预测曲线
plt.plot(range(n_output), y_pred_cnn_lstm_original, label="CNN+LSTM预测值", marker='o')

# 添加Excel表中的数据
plt.plot(excel_data['预测值'], label="LSTM预测值", marker='o')
plt.plot(excel_data['真实值'], label="真实值", marker='o')
plt.legend()

plt.tight_layout()
plt.show()
