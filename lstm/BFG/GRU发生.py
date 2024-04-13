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
bfgfasheng = scaler.fit_transform(bfgfasheng.values.reshape(-1, 1))
scaler_bfgfasheng = MinMaxScaler() # 专门用于bfgfasheng的反归一化
xiaohao = scaler_bfgfasheng.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 和你的 X 数据中时间步数一致
n_features =4       # 你输入数据中的特征数量，这里以 4 个为例
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
# 定义模型
model_gru = Sequential()
model_gru.add(GRU(units=128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(GRU(units=128, activation='tanh', return_sequences=True))  # 添加一个额外的GRU层，保持返回序列
model_gru.add(GRU(units=128, activation='tanh'))  # 最后一层GRU，不返回序列
model_gru.add(Dense(units=n_output))  # 输出层神经元数量为10，用于预测10个点
model_gru.compile(optimizer='adam', loss='mse')

# 注意 X 和 y 的形状应该匹配模型的输入和输出要求
# 训练模型
history = model_gru.fit(X, y, epochs=100, batch_size=32, validation_split=0.15)

y_pred = model_gru.predict(X_test)

 #反归一化预测值和真实值
y_pred_original = scaler_bfgfasheng.inverse_transform(y_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:n_output]

# 可视化比较
plt.plot(range(n_output), y_true_original, label="真实值",marker='o')
plt.plot(range(n_output), y_pred_original, label="预测值",marker='o')
# 选择最后30个点
last_30_true = y_true_original.flatten()[-30:]
last_30_pred = y_pred_original.flatten()[-30:]

# 创建DataFrame保存最后30个点的预测结果
df_result = pd.DataFrame({'真实值': last_30_true, '预测值': last_30_pred})

# 将DataFrame保存为Excel文件
df_result.to_excel('GRUBFG发生.xlsx', index=False)

plt.legend()
plt.show()
# 重塑 y_test 和 y_pred，使它们维度相同
y_test_reshaped = y_test.reshape(-1, n_output)
y_pred_reshaped = y_pred.reshape(-1, n_output)

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test_reshaped, y_pred_reshaped)
rmse = math.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
# 计算绝对百分比误差（APE）
APE = np.abs((y_true_original - y_pred_original) / y_true_original) * 100

# 计算MAPE
MAPE = np.mean(APE)
print('MAPE:', MAPE)
print('MSE:', mse)
print('RMSE:', rmse)
# 创建包含指标的 DataFrame
metrics_data = {
    '指标': ['MAPE', 'MSE', 'RMSE'],
    '值': [MAPE, mse, rmse]
}
df_metrics = pd.DataFrame(metrics_data)

# 将 DataFrame 保存为 Excel 文件
df_metrics.to_excel('GRUBFG发生数据.xlsx', index=False)
