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
l_i1jt = {}
l1_value = {}
l_i2jt = {}
l2_value = {}
V_jt = {}
F_value = {}
V_value = {}
hv_i1_min = {}
hv_i2_min = {}
h_jt = {}
max_var = {}
hve = {}
H_value = {}
e = {}
e_value ={}
bodong = {}
H_dual = {}
tidu1 = {}
e_total = {}
lamda_dual = {}
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
hv_i2_min = [1469252000, 8472000, 2988180000, 5008080, 3502520, 16696400, 19175940, 19175940, 5007923.2, 879740000, 836000000]
hv_i2_max = [2938504000, 11296000, 3386604000, 5723520, 4002880, 19081600, 21915360, 21915360, 5723340.8, 945720500, 1003200000]
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
initial_h_values = [200000, 80000, 150000]
mu1 = {}
for j in range(J):
    for t in range(1, T):
        mu1[j,t] = 0
def child_solution1():
    model = gp.Model()
    for j in range(J):
           for t in range(1,T):
                  F_jt[j, t] = model.addVar(name=f"F_{j}_{t}")
    objective_expr = gp.LinExpr()
    for j in range(J):
        for t in range(1, T):
            objective_expr += F_jt[j, t]*c_jf[j]
            objective_expr -=mu1[j,t]*F_jt[j, t]
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for j in range(J):
            F_var = F_jt[j,t]
            F_value[j,t] = F_var.x  # 放散（除了给电厂的其他的都来了）
    objective_value = objective_expr.getValue()

    return F_value,objective_value
def child_solution2():
    model = gp.Model()
    for j in range(J):
           for t in range(1,T):
                V_jt[j, t] = model.addVar(name=f"V_jt_{j}_{t}")
    # 电厂热值约束
    for t in range(1, T):
        lhs_heat = gp.LinExpr()
        for j in range(J):
            lhs_heat.addTerms([R_j[j]], [V_jt[j, t]])
        model.addConstr(lhs_heat <= 50000 * 8243)
    for t in range(1, T):
        for j in range(J):
            if j == 2:
                model.addConstr(V_jt[j, t] == 0)
            else:
                model.addConstr(V_jt[0, t] - 9 * V_jt[1, t] == 0)
    objective_expr = gp.LinExpr()
    hest = gp.LinExpr()
    for t in range(1,T):
        hest = 0
        for j in range(J):
            hest += V_jt[j, t] * R_j[j]
        objective_expr += (50000 - hest / 8243) * 0.86
        objective_expr -=mu1[j,t]*V_jt[j, t]
    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for j in range(J):
            V_var = V_jt[j,t]
            V_value[j,t] = V_var.x  # 发电厂煤气量（很固定）
    objective_value = objective_expr.getValue()
    return V_value,objective_value

def child_solution3():
    model = gp.Model()
    for i1 in range(I1):
        for j in range(J):
            for t in range(T):
                l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
    for i1 in range(I1):
        for t in range(1, T):
            for j in range(J):
                if j == jj[i1]:
                    model.addConstr(l_i1jt[i1, j, t] * R_j[j] >= hv_i1_min[i1])
                    model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_max[i1])
                else:
                    model.addConstr(l_i1jt[i1, j, t] == 0)
    objective_expr = gp.LinExpr()
    for i1 in range(I1):
        for t in range(1, T):
            for j in range(J):
                objective_expr -=mu1[j,t]*l_i1jt[i1, j, t]
    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for i1 in range(I1):
            for j in range(J):
                l1_var = l_i1jt[i1, j,t]
                l1_value[i1,j, t] = l1_var.x  # 单一煤气消耗量（正常）
    objective_value = objective_expr.getValue()

    return l1_value,objective_value
def child_solution4():
    model = gp.Model()
    for i2 in range(I2):
        for j in range(J):
            for t in range(T):
                L1 = L[i2]
                for l in range(L1):
                    l_i2jt[i2, j, t, l] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{j}_{t}_{l}", lb=0)
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            for l in range(L1):
                e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
    # 混合比例约束
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
                lll.addTerms(value2, e[i, l, t], l_i2jt[i, 0, t, l])
                lll.addTerms(-value1, e[i, l, t], l_i2jt[i, 1, t, l])
                lll1.addTerms(value3, e[i, l, t], l_i2jt[i, 0, t, l])
                lll1.addTerms(-value1, e[i, l, t], l_i2jt[i, 2, t, l])
                lll2.addTerms(value3, e[i, l, t], l_i2jt[i, 1, t, l])
                lll2.addTerms(-value2, e[i, l, t], l_i2jt[i, 2, t, l])
                model.addConstr(lll == 0)
                model.addConstr(lll1 == 0)
                model.addConstr(lll2 == 0)
    # #只能有一个方案约束
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            model.addConstr(gp.quicksum(e[i, l, t] for l in range(L1)) == 1)
    for i2 in range(I2):
        for t in range(1, T):
            lhs_heat2 = gp.QuadExpr()
            for j in range(J):
                L1 = L[i2]
                for l in range(L1):
                    lhs_heat2 += R_j[j] * l_i2jt[i2, j, t, l]*e[i2, l, t]
            rhs_heat_min2 = hv_i2_min[i2]  # 最小热量值
            rhs_heat_max2 = hv_i2_max[i2]  # 最大热量值
            model.addConstr(lhs_heat2 >= rhs_heat_min2)
            model.addConstr(lhs_heat2 <= rhs_heat_max2)
    objective_expr = gp.QuadExpr()
    for j in range(J):
        for t in range(1,T):
            for i2 in range(I2):
                L1 = L[i2]
                for l in range(L1):
                    objective_expr.addTerms([-mu1[j,t]], [l_i2jt[i2, j, t, l]], [e[i2, l, t]])
    # 设置最小化目标
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    for t in range(1, T):
        for i2 in range(I2):
            for j in range(J):
                L1 = L[i]
                for l in range(L1):
                    l2_var = l_i2jt[i2, j, t, l]
                    l2_value[i2, j, t, l] = l2_var.x  # 混合煤气用户消耗量
    objective_value = objective_expr.getValue()
    return l2_value,objective_value
max_iter = 10
for iter in range(max_iter):
    F,obj1 = child_solution1()
    V,obj2 = child_solution2()
    l1,obj3 = child_solution3()
    l2,obj4 = child_solution4()
    obj = obj1+obj2+obj3+obj4
    for j in range(J):
        for t in range(1, T):
            obj += mu1[j,t]*f_j[j]*S_jt[j,t]
    print(obj)



