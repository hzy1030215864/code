import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

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

# 数据准备
scaler = MinMaxScaler()
fengya = scaler.fit_transform(fengya.values.reshape(-1, 1))
fl = scaler.fit_transform(fl.values.reshape(-1, 1))
fy = scaler.fit_transform(fy.values.reshape(-1, 1))
xiaohao = scaler.fit_transform(xiaohao.values.reshape(-1, 1))
scaler_bfgfasheng = MinMaxScaler()
bfgfasheng = scaler_bfgfasheng.fit_transform(bfgfasheng.values.reshape(-1, 1))

# 创建序列
sequence_length = 90
n_features = 4
n_output = 30
X, y = [], []
for i in range(len(xiaohao) - sequence_length - n_output):
    X.append(np.c_[xiaohao[i:i + sequence_length], fengya[i:i + sequence_length], fl[i:i + sequence_length], fy[i:i + sequence_length]])
    y.append(bfgfasheng[i + sequence_length:i + sequence_length + n_output])

X, y = np.array(X), np.array(y)

# 训练集和测试集划分
split = 0.8
split_index = int(len(X) * split)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 创建XGBoost回归模型
xgboost_model = XGBRegressor(objective='reg:squarederror',
                             learning_rate=0.6,       # 尝试不同的学习率
                             n_estimators=700,        # 尝试不同的树的数量
                             max_depth=8,             # 尝试不同的树的深度
                             min_child_weight=1,      # 尝试不同的最小叶子节点样本数
                             subsample=0.8,           # 尝试不同的行抽样比例
                             colsample_bytree=0.8,    # 尝试不同的列抽样比例
                             reg_alpha=0.2,           # 尝试不同的L1正则化参数
                             reg_lambda=0.1)          # 尝试不同的L2正则化参数
# 训练模型
xgboost_model.fit(X_train.reshape((X_train.shape[0], -1)), y_train.reshape((y_train.shape[0], -1)))

# 获取XGBoost模型的预测结果
xgboost_pred = xgboost_model.predict(X_test.reshape((X_test.shape[0], -1)))
xgboost_pred = xgboost_pred.reshape((xgboost_pred.shape[0], -1, n_output))

# 反归一化预测值和真实值
xgboost_pred_original = scaler_bfgfasheng.inverse_transform(xgboost_pred.reshape(-1, 1))[:n_output]
y_true_original = scaler_bfgfasheng.inverse_transform(y_test.reshape(-1, 1))[:n_output]

# 绘制预测结果和真实值的图形
plt.plot(range(n_output), y_true_original, label="真实值", marker='o')
plt.plot(range(n_output), xgboost_pred_original, label="XGBoost预测值", marker='x')
plt.legend()
plt.show()

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test.reshape(-1, n_output), xgboost_pred.reshape(-1, n_output))
rmse = sqrt(mse)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAPE:', mape(y_true_original, xgboost_pred_original))
