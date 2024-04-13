import math
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载数据

xiaohao_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\COG\轧钢混合系统COG.csv')
xiaohao = xiaohao_data["value"][1:10000]
# 数据准备
scaler_bfgfasheng = MinMaxScaler()  # 专门用于bfgfasheng的反归一化
xiaohao = scaler_bfgfasheng.fit_transform(xiaohao.values.reshape(-1, 1))
sequence_length = 90  # 输入序列的长度
output_length = 30  # 输出序列的长度
X, y = [], []
sequence_length = int(sequence_length)  # 转换为整数
output_length = int(output_length)  # 转换为整数

for i in range(len(xiaohao) - sequence_length - output_length):
    X.append(np.c_[xiaohao[i:i + sequence_length]])
    y.append(xiaohao[i + sequence_length:i + sequence_length + output_length])

X, y = np.array(X), np.array(y)
# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

def build_lstm_model(input_shape, units):
    model = Sequential()
    model.add(LSTM(units=units,  activation='tanh',input_shape=input_shape,return_sequences=True))
    model.add(LSTM(units=units,  activation='tanh',return_sequences=True))
    model.add(LSTM(units=units, activation='tanh',))
    model.add(Dense(units=30))  # 这里的 output_length 需要提前定义
    model.compile(optimizer='adam', loss='mse')
    return model
def create_dataset(X, y, time_window):
    Xs, ys = [], []
    for i in range(len(X) - time_window):
        Xs.append(X[i:(i + time_window)].reshape(time_window, -1))  # Reshape to 2D
        ys.append(y[i + time_window])
    return np.array(Xs), np.array(ys)

def evaluate(params):
    time_window, batch_size, hidden_units = map(int, params)
    units = int(hidden_units)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_reshaped, y_train_reshaped = create_dataset(X_train, y_train, time_window)
    X_test_reshaped, y_test_reshaped = create_dataset(X_test, y_test, time_window)

    model = build_lstm_model(input_shape=(time_window, X_train_reshaped.shape[2]), units=hidden_units)

    model.fit(X_train_reshaped, y_train_reshaped, epochs=100, batch_size=int(batch_size), validation_split=0.15)

    predictions = model.predict(X_test_reshaped)
    print("y_test_reshaped shape:", y_test_reshaped.shape)
    print("predictions shape:", predictions.shape)
    y_test_reshaped = y_test_reshaped.squeeze(axis=-1)

    mse = mean_squared_error(y_test_reshaped, predictions)
    print("mse shape:", mse.shape)

    # 返回均方误差的均值
    return np.mean(mse)

# 定义参数范围
param_bounds = [(1, 10), (32, 90), (50, 150)]  # 分别是时间窗口、批量大小、单元数的范围

# 运行差分进化算法进行优化
result = differential_evolution(evaluate, bounds=param_bounds, maxiter=50, popsize=10, disp=True)

# 打印最优参数和对应的适应度值
best_params = result.x
best_fitness = result.fun
print("Best Parameters:", best_params)
print("Best Fitness:", best_fitness)
best_units = best_params[2]  # 假设你的最佳参数中包含了LSTM的隐藏单元数
time_window, batch_size, hidden_units = map(int, best_params)
units = int(best_units)

best_model = build_lstm_model(input_shape=(sequence_length, X_train.shape[2]), units=units)

best_model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=1)

# 在训练后进行预测
y_pred = best_model.predict(X_test)
# 反归一化预测值和真实值
y_pred_original = scaler_bfgfasheng.inverse_transform(y_pred.reshape(-1, 1))[:output_length]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:output_length]

# 可视化比较
plt.plot(range(output_length), y_true_original, label="真实值",marker='o')
plt.plot(range(output_length), y_pred_original, label="预测值", marker='x')
plt.title('DE+LSTMCOG消耗')
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