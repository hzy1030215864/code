import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
xiaohao_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\COG\轧钢混合系统COG.csv')

# 提取需要的时间序列数据
xiaohao = xiaohao_data["value"][1:10000]

# 数据归一化
scaler = MinMaxScaler()
xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))

# 创建序列
sequence_length = 90
n_output = 30
X, y = [], []
for i in range(len(xiaohao) - sequence_length - n_output):
    X.append(np.c_[xiaohao[i:i + sequence_length]])
    y.append(xiaohao[i + sequence_length:i + sequence_length + n_output])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# param_grid = {
#     'n_estimators': [150],  # 注意这里用列表包裹参数值
#     'learning_rate': [0.15],
#     'max_depth': [3],
#     'min_child_weight': [2],
#     'subsample': [0.9],
#     'colsample_bytree': [0.9],
#     'gamma': [0.1],
#     'reg_alpha': [0.1],
#     'reg_lambda': [0.1]
# }

# 创建XGBoost回归模型，并设置学习率和树的数量
# 创建XGBoost回归模型，并设置不同的参数
xgboost_model = XGBRegressor(objective='reg:squarederror',
                             learning_rate=0.3,       # 尝试不同的学习率
                             n_estimators=500,        # 尝试不同的树的数量
                             max_depth=6,             # 尝试不同的树的深度
                             min_child_weight=1,      # 尝试不同的最小叶子节点样本数
                             subsample=0.8,           # 尝试不同的行抽样比例
                             colsample_bytree=0.8,    # 尝试不同的列抽样比例
                             reg_alpha=0.2,           # 尝试不同的L1正则化参数
                             reg_lambda=0.1)          # 尝试不同的L2正则化参数

# # 使用Grid Search进行参数搜索
# grid_search = GridSearchCV(xgboost_model, param_grid, scoring='neg_mean_squared_error', cv=3)
# grid_search.fit(X_train.reshape((X_train.shape[0], -1)), y_train.reshape((y_train.shape[0], -1)))

xgboost_model.fit(X_train.reshape((X_train.shape[0], -1)), y_train.reshape((y_train.shape[0], -1)))

# 获取XGBoost模型的预测结果
xgboost_pred = xgboost_model.predict(X_test.reshape((X_test.shape[0], -1)))
xgboost_pred = xgboost_pred.reshape((xgboost_pred.shape[0], -1, n_output))


# 结合XGBoost和CNN的预测结果
combined_pred = xgboost_pred

# 反归一化预测值和真实值
combined_pred_original = scaler.inverse_transform(combined_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler.inverse_transform(y_test.reshape(-1, 1))[:n_output]

# 绘制预测结果和真实值的图形
plt.plot(range(n_output), y_true_original, label="真实值", marker='o')
plt.plot(range(n_output), combined_pred_original, label="预测值", marker='x')
plt.legend()
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 计算MSE
mse = mean_squared_error(y_true_original, combined_pred_original)
print("均方误差（MSE）: {:.4f}".format(mse))

# 计算MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true_original, combined_pred_original)
print("平均绝对百分比误差（MAPE）: {:.4f}%".format(mape))

plt.show()
