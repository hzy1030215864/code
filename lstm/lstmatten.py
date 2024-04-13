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

xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))
scaler_bfgfasheng = MinMaxScaler() # 专门用于bfgfasheng的反归一化
bfgfasheng = scaler_bfgfasheng.fit_transform(bfgfasheng.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 输入序列的长度
output_length = 30  # 输出序列的长度
X, y = [], []
for i in range(len(xiaohao) - sequence_length - output_length):
    X.append(np.c_[xiaohao[i:i+sequence_length], fengya[i:i+sequence_length], fl[i:i+sequence_length], fy[i:i+sequence_length]])
    y.append(bfgfasheng[i+sequence_length:i+sequence_length+output_length])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
model = Sequential()

# 卷积层用于特征提取
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 4)))
model.add(MaxPooling1D(pool_size=2))

# 不要Flatten数据

# LSTM 层用于解析时间序列信息
# 注意：Conv1D后的MaxPooling1D会改变时间步的数量，所以向LSTM层的输入形状是变化的
model.add(LSTM(50, activation='relu'))

# 输出层
model.add(Dense(output_length))

# 模型编译
model.compile(optimizer='adam', loss='mse')

# 模型摘要
model.summary()
# 定义早停
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 模型检查点
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)


# 进行预测
y_pred = model.predict(X_test)

 #反归一化预测值和真实值
y_pred_original = scaler_bfgfasheng.inverse_transform(y_pred.reshape(-1, 1))[:output_length]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:output_length]

# 可视化比较
plt.plot(range(output_length), y_true_original, label="真实值")
plt.plot(range(output_length), y_pred_original, label="预测值")
plt.title('CNN+LSTM')

plt.legend()
plt.show()
# 重塑 y_test 和 y_pred，使它们维度相同
y_test_reshaped = y_test.reshape(-1, output_length)
y_pred_reshaped = y_pred.reshape(-1, output_length)

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test_reshaped, y_pred_reshaped)
rmse = math.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
print('MSE:', mse)
print('RMSE:', rmse)