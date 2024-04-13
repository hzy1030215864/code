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
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.model_selection import TimeSeriesSplit
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
epochs = 10 # 设置你想要的总轮数

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
xiaohao  = scaler_bfgfasheng.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 90  # 输入序列的长度
output_length = 30  # 输出序列的长度
X, y = [], []
for i in range(len(xiaohao) - sequence_length - output_length):
    X.append(np.c_[bfgfasheng[i:i+sequence_length], fengya[i:i+sequence_length], fl[i:i+sequence_length], fy[i:i+sequence_length]])
    y.append(xiaohao[i+sequence_length:i+sequence_length+output_length])

X, y = np.array(X), np.array(y)
# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# 伪代码：FA优化函数

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def attractiveness(beta0, gamma, distance):
    return beta0 * np.exp(-gamma * (distance ** 2))


# FA算法主体函数

def firefly_algorithm(objective_func, lower_bound, upper_bound, dim, n_fireflies=5, max_gen=10, alpha=0.5, beta0=1,
                      gamma=1):
    # 初始化萤火虫位置
    print("Optimization start")
    fireflies = np.random.randint(low=lower_bound, high=upper_bound , size=(n_fireflies, dim))
    print("Optimization start2")
    # 计算每个萤火虫的光强度
    light_intensity = np.zeros(n_fireflies)
    for i in range(n_fireflies):
        light_intensity[i] = objective_func(fireflies[i])

    print("Optimization start3")

    # 初始化最优解
    best_firefly = fireflies[np.argmin(light_intensity)]
    best_intensity = np.min(light_intensity)

    for gen in range(max_gen):
        print("Generation:", gen)

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                # 如果发现更亮的萤火虫，则向它移动
                if light_intensity[j] < light_intensity[i]:  # 最小化问题，寻找更小的光强度
                    distance = euclidean_distance(fireflies[i], fireflies[j])
                    beta = attractiveness(beta0, gamma, distance)
                    fireflies[i] += np.round(
                        beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(dim) - 0.5)).astype(int)
                    fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)  # 保持在边界内

                    # 评估新的解并更新光强
                    new_intensity = objective_func(fireflies[i])
                    if new_intensity < light_intensity[i]:
                        light_intensity[i] = new_intensity

                        # 如果当前解更好，更新最优解
                        if new_intensity < best_intensity:
                            best_intensity = new_intensity
                            best_firefly = fireflies[i]
                    # 随机化步长alpha，可以是固定的或者随着迭代次数减小
                alpha *= 0.97  # 例如，每次迭代步长减小3%

                # 添加迭代停止条件
                if gen == max_gen - 1:
                    break



    return best_firefly
# LSTM模型构建函数

def build_lstm_model(input_shape, units):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    model.add(Dense(units=output_length))  # 这里的 output_length 需要提前定义
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_dataset(X, y, time_window):
    Xs, ys = [], []
    for i in range(len(X) - time_window):
        Xs.append(X[i:(i + time_window)].reshape(time_window, -1))  # Reshape to 2D
        ys.append(y[i + time_window])
    return np.array(Xs), np.array(ys)
def lstm_objective(params):
    # 解包参数
    time_window, batch_size, hidden_units = map(int, params)
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集上的损失
        patience=5,  # 在5个epoch内如果没有改善则停止
        min_delta=1e-4,  # 被认为是改善的最小变化量
        restore_best_weights=True,  # 恢复最佳模型的权重
        verbose=1  # 打印详细信息
    )
    X_train_reshaped, y_train_reshaped = create_dataset(X_train, y_train, time_window)
    X_test_reshaped, y_test_reshaped = create_dataset(X_test, y_test, time_window)

    # 构建LSTM模型
    model = build_lstm_model(input_shape=(time_window, X_train_reshaped.shape[2]), units=hidden_units)

    # 使用TimeSeriesSplit进行时间序列的交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    cvscores = []
    for epoch in range(epochs):  # 假设你在代码中定义了 epochs 变量
        print(epoch)
        for train_index, val_index in tscv.split(X_train_reshaped):
            X_cv_train, X_cv_val = X_train_reshaped[train_index], X_train_reshaped[val_index]
            y_cv_train, y_cv_val = y_train_reshaped[train_index], y_train_reshaped[val_index]

            # 添加早期停止条件，确保指定了验证数据
            history = model.fit(
                X_cv_train,
                y_cv_train,
                validation_data=(X_cv_val, y_cv_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            y_pred = model.predict(X_cv_val)
            y_pred = y_pred.reshape(y_cv_val.shape[0], -1)
            y_cv_val = y_cv_val.reshape(y_cv_val.shape[0], -1)
            mse = mean_squared_error(y_cv_val, y_pred)
            cvscores.append(mse)

    mean_mse = np.mean(cvscores)
    print("Mean MSE:", mean_mse)

    return mean_mse

# 定义超参数的搜索范围
lower_bound = [1, 32, 10]  # 最小时间窗口，最小批量大小，最少单元数
upper_bound = [10, 128, 100]  # 最大时间窗口，最大批量大小，最多单元数
dim = 3  # 优化维度

# 使用FA寻找最优超参数
best_params = firefly_algorithm(
    lstm_objective,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    dim=dim
)
# 训练模型
best_units = best_params[2]  # 假设你的最佳参数中包含了LSTM的隐藏单元数
best_model = build_lstm_model(input_shape=(sequence_length, X_train.shape[2]), units=best_units)
best_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 在训练后进行预测
y_pred = best_model.predict(X_test)
# 反归一化预测值和真实值
y_pred_original = scaler_bfgfasheng.inverse_transform(y_pred.reshape(-1, 1))[:output_length]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:output_length]

# 可视化比较
plt.plot(range(output_length), y_true_original, label="真实值")
plt.plot(range(output_length), y_pred_original, label="预测值")
plt.title('LSTM')
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