import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from keras.losses import mean_squared_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense

# 定义遗传算法的优化问题
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 优化问题的参数范围
lower_bound = [1, 32, 10]  # 最小时间窗口，最小批量大小，最少单元数
upper_bound = [10, 128, 100]  # 最大时间窗口，最大批量大小，最多单元数
dim = 3  # 优化维度

# 创建遗传算法工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, lower_bound, upper_bound)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def create_dataset(X, y, time_window):
    Xs, ys = [], []
    for i in range(len(X) - time_window):
        Xs.append(X[i:(i + time_window)].reshape(time_window, -1))
        ys.append(y[i + time_window])
    return np.array(Xs), np.array(ys)

# 定义适应度函数
def evaluate(individual):
    # 解码超参数
    time_window, batch_size, units = map(float, individual[0])
    units = int(units)
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
    scaler_bfgfasheng = MinMaxScaler()  # 专门用于bfgfasheng的反归一化
    xiaohao = scaler_bfgfasheng.fit_transform(xiaohao.values.reshape(-1, 1))
    sequence_length = 90  # 输入序列的长度
    output_length = 30  # 输出序列的长度
    X, y = [], []
    sequence_length = int(sequence_length)  # 转换为整数
    output_length = int(output_length)  # 转换为整数

    for i in range(len(xiaohao) - sequence_length - output_length):
        X.append(np.c_[bfgfasheng[i:i + sequence_length], fengya[i:i + sequence_length], fl[i:i + sequence_length], fy[
                                                                                                                    i:i + sequence_length]])
        y.append(xiaohao[i + sequence_length:i + sequence_length + output_length])

    X, y = np.array(X), np.array(y)
    # 训练集和测试集划分
    split = 0.8
    split_index = int(len(X) * split)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    time_window = int(time_window)

    X_train_reshaped, y_train_reshaped = create_dataset(X_train, y_train, time_window)
    X_test_reshaped, y_test_reshaped = create_dataset(X_test, y_test, time_window)

    # 将 X_train_reshaped 和 X_test_reshaped 调整为三维
    X_train_reshaped = X_train_reshaped.reshape(X_train_reshaped.shape[0], time_window, -1)
    X_test_reshaped = X_test_reshaped.reshape(X_test_reshaped.shape[0], time_window, -1)

    print("X_train_reshaped shape before LSTM layer:", X_train_reshaped.shape)

    def build_model(units=units, time_window=time_window, input_dim=1):
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(time_window, input_dim)))
        model.add(Dense(output_length))
        model.compile(optimizer='adam', loss='mse')
        return model

    all_pipelines = []

    # 训练模型
    pipelines = []
    # ...
    # 在循环内部
    for i in range(X_train_reshaped.shape[2]):
        X_train_reshaped_i = X_train_reshaped[:, :, i]  # 获取当前特征的数据
        y_train_reshaped_i = y_train_reshaped[:, i].reshape(-1, 1)

        print("X_train_reshaped_i shape:", X_train_reshaped_i.shape)
        print("y_train_reshaped_i shape:", y_train_reshaped_i.shape)

        # 其他代码...

        # 创建单个 pipeline
        pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('model', KerasRegressor(build_fn=build_model, epochs=10, batch_size=int(batch_size), verbose=0))
        ])

        # 拟合当前特征的 scaler
        pipeline['scaler'].fit(X_train_reshaped_i)

        # 添加到 all_pipelines
        all_pipelines.append((f'pipeline_{i}', pipeline, [i]))

    # 创建 FeatureUnion
    full_pipeline = FeatureUnion(all_pipelines)

    # 将 FeatureUnion 添加到整体 pipeline
    full_pipeline = Pipeline([
        ('feature_union', full_pipeline),
        ('model', KerasRegressor(build_fn=build_model, epochs=10, batch_size=int(batch_size), verbose=0))
    ])

    # 拟合整体 pipeline
    full_pipeline.fit(X_train_reshaped, y_train_reshaped)

    # 进行预测并评估模型
    predictions = full_pipeline.predict(X_test_reshaped)
    for i, pipeline in enumerate(pipelines):
        X_test_reshaped_i = X_test_reshaped[:, :, i]  # 获取当前特征的测试数据
        pipeline['scaler'].transform(X_test_reshaped_i)  # 使用当前特征的 scaler 进行转换
        predictions_i = pipeline.predict(X_test_reshaped_i)
        predictions[:, i] = predictions_i.flatten()

    # ...

    # ...

    score = mean_squared_error(y_test_reshaped, predictions)

    # 打印评估分数
    print("Model Score:", score)

    return -score,  # Minimize MSE

# 注册遗传算法所需的操作
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建初始种群
population = toolbox.population(n=10)

# 运行遗传算法进行优化
algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=20, cxpb=0.7, mutpb=0.3, ngen=10, stats=None,
                          halloffame=None, verbose=True)

# 打印最终的最优个体和适应度
best_individual = tools.selBest(population, k=1)[0]
best_fitness = evaluate(best_individual)[0]
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)
