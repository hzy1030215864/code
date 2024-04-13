import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
result_df = pd.DataFrame(columns=["Iteration", "Variable_Values"])
loaded_df = pd.read_excel("S_jt.xlsx")

# 将数据转换回矩阵（NumPy数组）
S_jt = loaded_df.to_numpy()
f_j = [1100,431,90]
c_jf = [0.037,0.034,0.095]
J = 3  # Number of J values
T = 8  # Number of T values
I1 = 10  # Number of I1 values
I2 = 4  # Number of I2 values
O = {}
F_jt = {}
l_i1jt = {}
l_i2jt = {}
erfa_j = [0.35,0.35,0.35]
jj = [1,1,1,1,1,1,1,0,1,0]
H_j1 = {0:250000,1:150000,2:250000}
H_j0 = {0:50000,1:30000,2:50000}
R_j = [3350,18820,8364]
h_jt = {}
OO = {}
gas_consumption_processes = {}
for j in range(J):
    for t in range(T):
        gas_consumption_processes[j,t] = 0
seita = {}
V_j =[1,1,1]
for j in range(J):
    for t in range(T):
        O[j, t] = f_j[j] * S_jt[j, t]
hv_i1_min = [84771000,88320,1430880,1000720,178200,1430835.2,65280000,9050000,7000000,1100000000]
hv_i1_max = [169542000,176640,2861760,2001440,243000,2861670.4,85280000,11050000,8000000,1300000000]
initial_allocations = np.eye(I1, J)
initial_h_values = np.array([150000, 80000, 150000])
delta_jt = {}
for j in range(J):
    h_jt[j, 0] =initial_h_values[j]
    for t in range(1, T):
        h_jt[j, t] = 0
def objective_function(l_i1jt):
    # 将l_i1jt的形状调整为正确的形状
    l_i1jt = l_i1jt.reshape((I1, J,T))
    total_cost = 0.0
    for j in range(J):
        for t in range(1,T):
            h_jt[j, t] = max(H_j0[j], min(H_j1[j], h_jt[j, t]))
            delta_jt[j,t] = O[j, t] + (h_jt[j, t - 1] - H_j0[j]) * V_j[j] - np.sum(l_i1jt[:, j,t])
            if delta_jt[j,t] > 0:
                if 0 <= delta_jt[j,t] <= (H_j1[j] - H_j0[j]) * V_j[j]:
                    h_jt[j,t] = delta_jt[j,t] / V_j[j] + H_j0[j]
                    F_jt[j,t] = 0
                elif delta_jt[j,t] >= (H_j1[j] - H_j0[j]) * V_j[j]:
                    h_jt[j,t] = H_j1[j]
                    F_jt[j,t] = delta_jt[j,t] - (H_j1[j] - H_j0[j]) * V_j[j]
    for j in range(J):
        for t in range(1,T):
             total_cost += c_jf[j] * F_jt[j,t]

    return total_cost
l_i1jt = np.zeros((I1, J,T))
# def h_jt_constraint(h_jt):
#     constraints = []
#     for j in range(J):
#         for t in range(T):
#             constraints.append(h_jt[j, t] - H_j0[j])
#             constraints.append(H_j1[j] - h_jt[j, t])
#     return constraints
from scipy.optimize import LinearConstraint
#
# h_jt_constraints = LinearConstraint(h_jt_constraint, 0, np.inf)
# 设置初始解
# def gas_storage_constraints(h_jt):
#     constraints = []
#     for j in range(J):
#         for t in range(T):
#             # 确保 h_jt 在上下限范围内
#             constraints.append(h_jt[j, t] >= H_j0[j])
#             constraints.append(h_jt[j, t] <= H_j1[j])
#     return constraints
lambda_i1jt1 = np.zeros((I1,J,T))  # 初始化为零
lambda_i1jt2 = np.zeros((I1,J,T))  # 初始化为零
# 定义拉格朗日函数
def lagrangian(x,lambda_i1jt1,lambda_i1jt2):
    # 计算拉格朗日函数值
    lagrangian_value = 0
    l_i1jt = x.reshape((I1, J, T))
    lagrangian_value = objective_function(l_i1jt)  # 目标函数
    l_i1jt = l_i1jt.reshape((I1, J, T))

    for i in range(I1):
        for j in range(J):
            for t in range(T):
                lagrangian_value += lambda_i1jt1[i,j,t] * (hv_i1_min[i] - R_j[j]*l_i1jt[i, j,t])
                lagrangian_value += lambda_i1jt2[i,j,t] * (R_j[j]*l_i1jt[i, j,t] - hv_i1_max[i])
    return lagrangian_value
# 定义迭代参数
max_iterations = 100  # 最大迭代次数
tolerance = 1e-6  # 收敛容忍度
step_size = 0.1
x0 = np.zeros(I1 * J * T)
iteration_data = []
converged = False
from scipy.optimize import Bounds
lower_bound = 0  # l_ijt 下界为0，即 l_ijt 不能为负数
upper_bound = np.inf  # l_ijt 上界为正无穷
bounds = Bounds([lower_bound] * (I1 * J * T), [upper_bound] * (I1 * J * T))
for iteration in range(max_iterations):
    constraints = []
    # for j in range(J):
    #     for t in range(T):
    #         for i1 in range(I1):
    #             if j != jj[i1]:
    #                 # 如果 j 不等于 jj[i1]，则强制 xijt 为 0
    #                 constraint = LinearConstraint(
    #                     A=np.zeros((1, I1 * J * T)),  # 生成全零约束矩阵
    #                     lb=0,  # 下界为0
    #                     ub=0  # 上界为0
    #                 )
    #                 constraints.append(constraint)

    constraints_sequence = tuple(constraints)
    # 继续使用 constraints 来进行优化
    result = minimize(lagrangian, x0, args=(lambda_i1jt1, lambda_i1jt2), constraints=constraints_sequence,bounds=bounds, method='SLSQP')
    l_i1jt = result.x
    iteration_df = pd.DataFrame({"Iteration": [iteration], "Variable_Values": [result.x]})
    result_df = pd.concat([result_df, iteration_df], ignore_index=True)
    l_i1jt = l_i1jt.reshape((I1, J,T))
    current_lagrangian_value = lagrangian(l_i1jt, lambda_i1jt1, lambda_i1jt2)
    if iteration > 0:
        lagrangian_change = abs(current_lagrangian_value - previous_lagrangian_value)
        # 如果变化小于容忍度，则认为已经收敛
        if lagrangian_change < tolerance:
            print(f"收敛于第 {iteration} 次迭代")
            converged = True  # 设置收敛标志为True，表示已经收敛
            break  # 跳出当前循环
        # 更新 previous_lagrangian_value
    previous_lagrangian_value = current_lagrangian_value
    print(f"第{iteration}次迭代")
    minimum_value = objective_function(l_i1jt)
    print("最小值：", minimum_value)
    # 2. 更新拉格朗日乘子
    for i in range(I1):
        for j in range(J):
            for t in range(T):
                constraint1 = hv_i1_min[i] - R_j[j] * l_i1jt[i, j,t]
                constraint2 = R_j[j] * l_i1jt[i, j,t] - hv_i1_max[i]
                lambda_i1jt1[i,j,t] += step_size * constraint1
                lambda_i1jt2[i,j,t] += step_size * constraint2
    if converged:
       break  # 如果已经收敛，则跳出整个大循环
result_df.to_excel("variable_values.xlsx", index=False)