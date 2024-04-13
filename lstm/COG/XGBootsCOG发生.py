import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
xiaohao_data = pd.read_csv(r'E:\新建文件夹\代码\lstm\data\COG\COGFSL.csv')

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
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [1, 2, 3],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 0.2],
#     'reg_lambda': [0, 0.1, 0.2]
# }
# 创建XGBoost回归模型，并设置学习率和树的数量
xgboost_model = XGBRegressor(objective='reg:squarederror',
                             learning_rate=0.3,       # 尝试不同的学习率
                             n_estimators=700,        # 尝试不同的树的数量
                             max_depth=7,             # 尝试不同的树的深度
                             min_child_weight=1,      # 尝试不同的最小叶子节点样本数
                             subsample=0.8,           # 尝试不同的行抽样比例
                             colsample_bytree=0.8,    # 尝试不同的列抽样比例
                             reg_alpha=0.2,           # 尝试不同的L1正则化参数
                             reg_lambda=0.1)          # 尝试不同的L2正则化参数
# # 使用Grid Search进行参数搜索
# grid_search = GridSearchCV(xgboost_model, param_grid, scoring='neg_mean_squared_error', cv=3)
# grid_search.fit(X_train.reshape((X_train.shape[0], -1)), y_train.reshape((y_train.shape[0], -1)))



# 将训练数据转为一维数组
X_train_flatten = X_train.reshape((X_train.shape[0], -1))
y_train_flatten = y_train.reshape((y_train.shape[0], -1))

# 将测试数据转为一维数组
X_test_flatten = X_test.reshape((X_test.shape[0], -1))

# 拟合XGBoost模型
xgboost_model.fit(X_train_flatten, y_train_flatten)

# 在测试集上做预测
y_pred_flatten = xgboost_model.predict(X_test_flatten)

# 将预测结果reshape回原来的形状
y_pred = y_pred_flatten.reshape((y_pred_flatten.shape[0], -1, n_output))

# 反归一化预测值和真实值
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler.inverse_transform(y_test.reshape(-1, 1))[:n_output]

# 绘制预测结果和真实值的图形
plt.plot(range(n_output), y_true_original, label="真实值", marker='o')
plt.plot(range(n_output), y_pred_original, label="预测值", marker='x')
plt.legend()

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test.reshape(-1, n_output), y_pred.reshape(-1, n_output))
rmse = sqrt(mse)
print('MSE:', mse)
print('RMSE:', rmse)

plt.show()
