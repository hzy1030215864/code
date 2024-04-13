import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=["Variable", "Value"])
loaded_df = pd.read_excel("E:\新建文件夹\代码\lg\data\S_jt_24.xlsx")
# 将数据转换回矩阵（NumPy数组）
S_jt = loaded_df.to_numpy()
loaded_df1 = pd.read_excel("E:\新建文件夹\代码\lg\data\S_it_24.xlsx")
# 将数据转换回矩阵（NumPy数组）
S_it = loaded_df1.to_numpy()
f_j = [1100, 431, 90]

J = 3  # Number of J values
T = 24  # Number of T values
I1 = 7  # Number of I1 values
I2 = 11  # Number of I2 values
L = [3, 1, 3, 1, 1, 2, 2, 2, 1, 4, 1]
O = {}
F_jt = {}
F_jt1 = {}
F_jt2 = {}
l_i1jt = {}
l_i2jt = {}
l_i1jt1 = {}
l_i2jt1 = {}
l_i1jt2 = {}
l_i2jt2 = {}
V_jt = {}
V_jt1 = {}
V_jt2 = {}
hv_i1_min = {}
hv_i2_min = {}
h_jt = {}
max_var = {}
hve = {}
e = {}
erfa_j = [0.35, 0.35, 0.35]
jj = [1, 1, 1, 1, 0, 1, 0]
c_jf = [0.037, 0.343, 0.095]
# 煤气柜上下线
H_j1 = {0: 240000, 1: 100000, 2: 207000}
H_j0 = {0: 140000, 1: 65000, 2: 50000}
danyi = [0, 4, 5, 8, 13, 16, 17]
hunhe = [1, 2, 3, 6, 7, 9, 10, 11, 12, 14, 15]
hv_i1_min = [84771000, 88155200, 309120, 210600, 1154020000, 65870000, 10050000]

hv_i1_max = [169542000, 100748800, 353280, 226800, 1236450000, 84690000, 10050000]
hv_i2_min = [1469252000, 8472000, 2988180000, 5008080, 3502520, 16696400, 19175940, 19175940, 5007923.2, 879740000,
             836000000]

hv_i2_max = [2938504000, 11296000, 3386604000, 5723520, 4002880, 19081600, 21915360, 21915360, 5723340.8, 945720500,
             1003200000]
# w = [56514,3673130,56480,1992120,62968,368,10840,35740,1620,13400,15390,15390,12560,8243,2199350,8360,18820,3350]
R_j = [3350, 18820, 8364]
erf = [[[1, 0, 0], [0, 1, 0], [0.95, 0.05, 0]],
       [[0.18, 0.82, 0]],
       [[1, 0, 0], [0.95, 0.05, 0], [0.83, 0.17, 0]],
       [[0.62, 0.38, 0]],
       [[0.62, 0.38, 0]],
       [[0.45, 0.35, 0.2], [0.6, 0.4, 0]],
       [[0.54, 0.26, 0.2], [0.66, 0.34, 0]],
       [[0.45, 0.38, 0.07], [0.56, 0.44, 0]],
       [[0.71, 0.29, 0]],
       [[0, 1, 0], [0, 0.1, 0.9], [0.65, 0.1, 0.25], [0, 0.35, 0.65]],
       [[0.68, 0.32, 0]]]
max_iterations = 10  # 最大迭代次数
lambda_i1jt1 = {}
lambda_i1jt2 = {}
lambda_i1jt3 = {}
lambda_i1jt4 = {}
lambda_i1jt5 = {}
sete = {}
l1_value = {}
l2_value = {}
F1_value = {}
e_value = {}
V_value = {}
l2_value1 = {}
F1_value1 = {}
e_value1 = {}
V_value1 = {}
l2_value2 = {}
F1_value2 = {}
e_value2 = {}
V_value2 = {}
tidu1 = np.zeros((I2, T))
tidu2 = np.zeros((I2, T))
tidu3 = np.zeros((I2, T))
tidu4 = np.zeros((I2, T))
tidu5 = np.zeros((I2, T))
for i in range(I2):
    for t in range(1, T):
        lambda_i1jt1[i, t] = 0
for i in range(I2):
    for t in range(1, T):
        lambda_i1jt2[i, t] = 0
for i in range(I2):
    for t in range(1, T):
        lambda_i1jt3[i, t] = 0
for i in range(I2):
    for t in range(1, T):
        lambda_i1jt4[i, t] = 0.1
for i in range(I2):
    for t in range(1, T):
        lambda_i1jt5[i, t] = 0.5
initial_h_values = [200000, 80000, 150000]
for j in range(J):
    for t in range(T):
        O[j, t] = f_j[j] * S_jt[j, t]


def solve_subproblem_1(lambda_i1jt1, lambda_i1jt2, lambda_i1jt4, lambda_i1jt5):
    model = gp.Model()
    for i1 in range(I1):
        for t in range(T):
            l_i1jt[i1, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{t}", lb=0)
    for i2 in range(I2):
        for t in range(T):
            l_i2jt[i2, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{t}", lb=0)
    for t in range(T):
        F_jt[t] = model.addVar(name=f"F_{t}")
        V_jt[t] = model.addVar(name=f"V_jt_{t}")
    for t in range(T):
        h_jt[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{t}", lb=0)
        hve[t] = model.addVar(name=f"hve_{t}")
    model.addConstr(h_jt[0] == initial_h_values[0])
    for t in range(1, T):
        lhs1 = gp.LinExpr()
        lhs1 = O[0, t] + h_jt[t - 1] - h_jt[t]
        for i1 in range(I1):
            lhs1.addTerms([-1], [l_i1jt[i1, t]])
        for i2 in range(I2):
            lhs1.addTerms([-1], [l_i2jt[i2, t]])
        lhs1.addTerms([-1], [V_jt[t]])
        lhs1.addTerms([-1], [F_jt[t]])
        model.addConstr(lhs1 == 0)
    for t in range(1, T):
        model.addConstr(h_jt[t] >= H_j0[0])
        model.addConstr(h_jt[t] <= H_j1[0])
    for i1 in range(I1):
        for t in range(1, T):
            if jj[i1] == 0:
                model.addConstr(l_i1jt[i1, t] * R_j[0] >= hv_i1_min[i1])
                model.addConstr(l_i1jt[i1, t] * R_j[0] <= hv_i1_max[i1])
            else:
                model.addConstr(l_i1jt[i1, t] == 0)
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            for l in range(L1):
                e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            model.addConstr(gp.quicksum(e[i, l, t] for l in range(L1)) == 1)
    for t in range(1, T):
        lhs_heat = gp.LinExpr()
        lhs_heat.addTerms([R_j[0]], [V_jt[t]])
        model.addConstr(lhs_heat <= 412150000)
        model.addConstr(lhs_heat >= 312450000)
    for i2 in range(I2):
        for t in range(1, T):
            L1 = L[i2]#每道工序的方案总数
            for l in range(L1):
                value1 = erf[i2][l][0]
                value2 = erf[i2][l][1]
                value3 = erf[i2][l][2]
                if value1 == 0:#判断该方案中高炉煤气配比为0
                    model.addConstr(l_i2jt[i2, t] == 0)
                else:
                    model.addConstr(l_i2jt[i2, t] * e[i2, l, t] * R_j[0] + l_i2jt[i2, t] * (value2 / value1) * e[i2, l, t] * R_j[1] + l_i2jt[i2, t] * (value3 / value1) * e[i2, l, t] * R_j[2]
                        >= hv_i2_min[i2] * e[i2, l, t])
                    model.addConstr(l_i2jt[i2, t] * e[i2, l, t] * R_j[0] + l_i2jt[i2, t] * (value2 / value1) * e[i2, l, t] * R_j[1] + l_i2jt[i2, t] * (value3 / value1) * e[i2, l, t] * R_j[2]
                        <= hv_i2_max[i2] * e[i2, l, t])
    z = {}
    for t in range(1, T):
        z[t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    for t in range(1, T):
        model.addConstr(h_jt[t] - h_jt[t - 1] <= z[t])
        model.addConstr(h_jt[t - 1] - h_jt[t] <= z[t])
    objective_expr = gp.QuadExpr()
    for t in range(1, T):
        hest = V_jt[t] * R_j[0]  # 第一种煤气的热量
        objective_expr += c_jf[0] * F_jt[t]  # 放散
        objective_expr += z[t] * c_jf[0]  # 煤气柜位波动
        objective_expr += (50000 - hest / 8243) * 0.86  # 热量不足导致的电力
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            lll = gp.QuadExpr()
            lll1 = gp.QuadExpr()
            for l in range(L1):
                value2 = erf[i][l][1]
                value3 = erf[i][l][2]
                lll.addTerms(value2, e[i, l, t], l_i2jt[i, t])
                lll1.addTerms(value3, e[i, l, t], l_i2jt[i, t])
            objective_expr += lambda_i1jt1[i, t] * (lll)
            objective_expr += lambda_i1jt2[i, t] * (lll1)

    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    # 清理模型
    for t in range(1, T):
        for i1 in range(I1):
            l1_var = l_i1jt[i1, t]
            l1_value[i1, t] = l1_var.x  # 单一煤气消耗量（正常）
    for t in range(1, T):
        for i2 in range(I2):

            l2_var = l_i2jt[i2, t]
            l2_value[i2, t] = l2_var.x  # 混合煤气用户消耗量
    for t in range(1, T):
        F1_var = F_jt[t]
        F1_value[t] = F1_var.x  # 放散（除了给电厂的其他的都来了）
        V_var = V_jt[t]
        V_value[t] = V_var.x  # 发电厂煤气量（很固定）
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            for l in range(L1):
                e_var = e[i, l, t]
                e_value[i, l, t] = e_var.x  # 方案
    objective_value = objective_expr.getValue()
    model.reset()
    return objective_value, l2_value, F1_value, e_value


def solve_subproblem_2(lambda_i1jt1, lambda_i1jt3, lambda_i1jt4, lambda_i1jt5):
    model = gp.Model()
    for i1 in range(I1):
        for t in range(T):
            l_i1jt1[i1, t] = model.addVar(name=f"l_i1_{i1}_{t}", vtype=gp.GRB.CONTINUOUS, lb=0)
    for i2 in range(I2):
        for t in range(T):
            l_i2jt1[i2, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2 }_{t}", lb=0)
    for t in range(T):
        F_jt1[t] = model.addVar(name=f"F_{t}")
        V_jt1[t] = model.addVar(name=f"V_jt_{t}")
    for t in range(T):
        h_jt[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
        hve[t] = model.addVar(name=f"hve_{t}")
    model.addConstr(h_jt[1] == initial_h_values[1])
    for t in range(1, T):
        lhs1 = gp.LinExpr()
        lhs1 = O[1, t] + h_jt[t - 1] - h_jt[t]
        for i1 in range(I1):
            lhs1.addTerms([-1], [l_i1jt1[i1, t]])
        for i2 in range(I2):
            lhs1.addTerms([-1], [l_i2jt1[i2, t]])
        lhs1.addTerms([-1], [F_jt1[t]])
        model.addConstr(lhs1 == 0)
    for t in range(1, T):
        model.addConstr(h_jt[t] >= H_j0[1])
        model.addConstr(h_jt[t] <= H_j1[1])
    for i1 in range(I1):
        for t in range(1, T):
            if jj[i1] == 1:
                model.addConstr(l_i1jt1[i1, t] * R_j[1] >= hv_i1_min[i1])
                model.addConstr(l_i1jt1[i1, t] * R_j[1] <= hv_i1_max[i1])
            else:
                model.addConstr(l_i1jt1[i1, t] == 0)
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            for l in range(L1):
                e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            model.addConstr(gp.quicksum(e[i, l, t] for l in range(L1)) == 1)
    for i2 in range(I2):
        for t in range(1, T):
            L1 = L[i2]
            for l in range(L1):
                value1 = erf[i2][l][0]
                value2 = erf[i2][l][1]
                value3 = erf[i2][l][2]
                if value2 == 0:
                    model.addConstr(l_i2jt1[i2, t] == 0)
                else:
                    model.addConstr(
                        l_i2jt1[i2, t] * e[i2, l, t] * R_j[1] + l_i2jt1[i2, t] * (value1 / value2) * e[i2, l, t] * R_j[
                            0] + l_i2jt1[i2, t] * (value3 / value2) * e[i2, l, t] * R_j[2] >= hv_i2_min[i2] * e[
                            i2, l, t])
                    model.addConstr(
                        l_i2jt1[i2, t] * e[i2, l, t] * R_j[1] + l_i2jt1[i2, t] * (value1 / value2) * e[i2, l, t] * R_j[
                            0] + l_i2jt1[i2, t] * (value3 / value2) * e[i2, l, t] * R_j[2] <= hv_i2_max[i2] * e[
                            i2, l, t])
    z = {}
    for t in range(1, T):
        z[t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    for t in range(1, T):
        model.addConstr(h_jt[t] - h_jt[t - 1] <= z[t])
        model.addConstr(h_jt[t - 1] - h_jt[t] <= z[t])
    model.update()
    objective_expr1 = gp.QuadExpr()
    for t in range(1, T):
        objective_expr1 += c_jf[1] * F_jt1[t]
        objective_expr1 += z[t] * c_jf[j]
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            lll = gp.QuadExpr()
            lll1 = gp.QuadExpr()
            lll2 = gp.QuadExpr()
            for l in range(L1):
                value1 = erf[i][l][0]
                value3 = erf[i][l][2]
                lll.addTerms(-value1, e[i, l, t], l_i2jt1[i, t])
                lll2.addTerms(value3, e[i, l, t], l_i2jt1[i, t])
            objective_expr1 += lambda_i1jt1[i, t] * (lll)
            objective_expr1 += lambda_i1jt3[i, t] * (lll2)

    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr1, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for i2 in range(I2):
            l2_var = l_i2jt1[i2, t]
            l2_value1[i2, t] = l2_var.x
    for t in range(1, T):
        F1_var = F_jt1[t]
        F1_value1[t] = F1_var.x
        V_var = V_jt1[t]
        V_value1[t] = V_var.x
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            for l in range(L1):
                e_var = e[i, l, t]
                e_value1[i, l, t] = e_var.x
    objective_value = objective_expr1.getValue()
    model.reset()

    return objective_value, l2_value1, F1_value1, e_value1


def solve_subproblem_3(lambda_i1jt2, lambda_i1jt3, lambda_i1jt4, lambda_i1jt5):
    model = gp.Model()
    for i1 in range(I1):
        for t in range(T):
            l_i1jt2[i1, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{t}", lb=0)
    for i2 in range(I2):
        for t in range(T):
            l_i2jt2[i2, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{t}", lb=0)
    for t in range(T):
        F_jt2[t] = model.addVar(name=f"F_{t}")
    for t in range(T):
        h_jt[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
        hve[t] = model.addVar(name=f"hve_{t}")
    model.addConstr(h_jt[2] == initial_h_values[2])
    for t in range(1, T):
        lhs1 = gp.LinExpr()
        lhs1 = O[2, t] + h_jt[t - 1] - h_jt[t]
        for i1 in range(I1):
            lhs1.addTerms([-1], [l_i1jt2[i1, t]])
        for i2 in range(I2):
            lhs1.addTerms([-1], [l_i2jt2[i2, t]])

        lhs1.addTerms([-1], [F_jt2[t]])
        model.addConstr(lhs1 == 0)
    for t in range(1, T):
        model.addConstr(h_jt[t] >= H_j0[0])
        model.addConstr(h_jt[t] <= H_j1[0])
    for i1 in range(I1):
        for t in range(1, T):
            if jj[i1] == 2:
                model.addConstr(l_i1jt2[i1, t] * R_j[2] >= hv_i1_min[i1])
                model.addConstr(l_i1jt2[i1, t] * R_j[2] <= hv_i1_max[i1])
            else:
                model.addConstr(l_i1jt2[i1, t] == 0)
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            for l in range(L1):
                e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            model.addConstr(gp.quicksum(e[i, l, t] for l in range(L1)) == 1)
    for i2 in range(I2):
        for t in range(1, T):
            L1 = L[i2]
            for l in range(L1):
                value1 = erf[i2][l][0]
                value2 = erf[i2][l][1]
                value3 = erf[i2][l][2]
                if value3 == 0:
                    model.addConstr(l_i2jt2[i2, t] == 0)
                else:
                    model.addConstr(
                        l_i2jt2[i2, t] * e[i2, l, t] * R_j[2] + l_i2jt2[i2, t] * (value1 / value3) * e[i2, l, t] * R_j[
                            0] + l_i2jt2[i2, t] * (value2 / value3) * e[i2, l, t] * R_j[1] >= hv_i2_min[i2] * e[
                            i2, l, t])
                    model.addConstr(
                        l_i2jt2[i2, t] * e[i2, l, t] * R_j[2] + l_i2jt2[i2, t] * (value1 / value3) * e[i2, l, t] * R_j[
                            0] + l_i2jt2[i2, t] * (value2 / value3) * e[i2, l, t] * R_j[1] <= hv_i2_max[i2] * e[
                            i2, l, t])
    z = {}
    for t in range(1, T):
        z[t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    for t in range(1, T):
        model.addConstr(h_jt[t] - h_jt[t - 1] <= z[t])
        model.addConstr(h_jt[t - 1] - h_jt[t] <= z[t])
    objective_expr = gp.QuadExpr()
    for t in range(1, T):
        objective_expr += c_jf[2] * F_jt2[t]
        objective_expr += z[t] * c_jf[2]
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            lll = gp.QuadExpr()
            lll1 = gp.QuadExpr()
            lll2 = gp.QuadExpr()
            for l in range(L1):
                value1 = erf[i][l][0]
                value2 = erf[i][l][1]
                value3 = erf[i][l][2]
                lll1.addTerms(-value1, e[i, l, t], l_i2jt2[i, t])
                lll2.addTerms(-value2, e[i, l, t], l_i2jt2[i, t])
            objective_expr += lambda_i1jt2[i, t] * (lll1)
            objective_expr += lambda_i1jt3[i, t] * (lll2)

    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for i2 in range(I2):
            l2_var = l_i2jt2[i2, t]
            l2_value2[i2, t] = l2_var.x
    for t in range(1, T):
        F2_var = F_jt2[t]
        F1_value2[t] = F2_var.x
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            for l in range(L1):
                e_var = e[i, l, t]
                e_value2[i, l, t] = e_var.x
    objective_value = objective_expr.getValue()
    model.reset()

    return objective_value, l2_value2, F1_value2, e_value2


def tidu(e1, l2_1, l2_2, l2_3):
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    for i in range(I2):
        for t in range(1, T):
            L1 = L[i]
            for l in range(L1):
                value1 = erf[i][l][0]
                value2 = erf[i][l][1]
                value3 = erf[i][l][2]
                t1 += value2 * e1[i, l, t] * l2_1[i, t] - value1 * e1[i, l, t] * l2_2[i, t]
                t2 += value3 * e1[i, l, t] * l2_1[i, t] - value1 * e1[i, l, t] * l2_3[i, t]
                t3 += value3 * e1[i, l, t] * l2_2[i, t] - value2 * e1[i, l, t] * l2_3[i, t]

            tidu1[i, t] = t1
            tidu2[i, t] = t2
            tidu3[i, t] = t3

    return tidu1, tidu2, tidu3


max_iterations = 10

for iteration in range(max_iterations):

    obj1, l2_1, F1, e1 = solve_subproblem_1(lambda_i1jt1, lambda_i1jt2, lambda_i1jt4, lambda_i1jt5)

    obj2, l2_2, F2, e2 = solve_subproblem_2(lambda_i1jt1, lambda_i1jt3, lambda_i1jt4, lambda_i1jt5)

    obj3, l2_3, F3, e3 = solve_subproblem_3(lambda_i1jt2, lambda_i1jt3, lambda_i1jt4, lambda_i1jt5)

    obj = obj1 + obj2 + obj3

    tidu1, tidu2, tidu3 = tidu(e1, l2_1, l2_2, l2_3)

    for i in range(I2):
        for t in range(1, T):
            if tidu1[i, t] == 0:
                lambda_i1jt1[i, t] = lambda_i1jt1[i, t]
            else:
                lambda_i1jt1[i, t] = [(462556.0 - obj) / (np.mat(tidu1[i, t]) * np.mat(tidu1[i, t]))] * np.mat(
                    tidu1[i, t])
            if tidu2[i, t] == 0:
                lambda_i1jt2[i, t] = lambda_i1jt2[i, t]
            else:
                lambda_i1jt2[i, t] = [(462556.0 - obj) / (np.mat(tidu2[i, t]) * np.mat(tidu2[i, t]))] * np.mat(
                    tidu2[i, t])
            if tidu3[i, t] == 0:
                lambda_i1jt3[i, t] = lambda_i1jt3[i, t]
            else:
                lambda_i1jt3[i, t] = [(462556.0 - obj) / (np.mat(tidu3[i, t]) * np.mat(tidu3[i, t]))] * np.mat(
                    tidu3[i, t])

    print(iteration)
    print(obj)
