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
model = gp.Model()
J = 3  # Number of J values
T = 24  # Number of T values
I1 = 7  # Number of I1 values
I2 = 11  # Number of I2 values
L = [3, 1, 3, 1, 1, 2, 2, 2, 1, 4, 1]
O = {}
F_jt = {}
l_i1jt = {}
l_i2jt = {}
V_jt = {}
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
initial_h_values = [200000, 80000, 150000]
# for i1 in range(I1):
#     i =danyi[i1]
#     for t in range(T):
#         hv_i1_min[i1,t]=w[i]*S_it[i,t]
# for i2 in range(I2):
#     i =hunhe[i2]
#     for t in range(T):
#         hv_i2_min[i2,t]=w[i]*S_it[i,t]
for i1 in range(I1):
    for j in range(J):
        for t in range(T):
            l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
for i2 in range(I2):
    for j in range(J):
        for t in range(T):
            L1 = L[i2]
            for l in range(L1):
                l_i2jt[i2, j, t, l] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{j}_{t}_{l}", lb=0)
for j in range(J):
    for t in range(T):
        O[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"O_{j}_{t}", lb=0)
        F_jt[j, t] = model.addVar(name=f"F_{j}_{t}")
        V_jt[j, t] = model.addVar(name=f"V_jt_{j}_{t}")
for j in range(J):
    for t in range(T):
        h_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{j}_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
        hve[j, t] = model.addVar(name=f"hve_{j}_{t}")
for t in range(1, T):
    for i in range(I2):
        L1 = L[i]
        for l in range(L1):
            e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
for j in range(J):
    for t in range(T):
        model.addConstr(O[j, t] == f_j[j] * S_jt[j, t])
for j in range(J):
    model.addConstr(h_jt[j, 0] == initial_h_values[j])
for j in range(J):
    for t in range(1, T):
        lhs1 = gp.QuadExpr()
        lhs1.addTerms([1], [O[j, t]])
        lhs1.addTerms([1], [h_jt[j, t - 1]])
        for i1 in range(I1):
            lhs1.addTerms([-1], [l_i1jt[i1, j, t]])
        for i2 in range(I2):
            L1 = L[i2]
            for l in range(L1):
                lhs1.addTerms([-1], [l_i2jt[i2, j, t, l]],[e[i2, l,t]])
        lhs1.addTerms([-1], [h_jt[j, t]])
        lhs1.addTerms([-1], [V_jt[j, t]])
        lhs1.addTerms([-1], [F_jt[j, t]])
        model.addConstr(lhs1 == 0)
for j in range(J):
    for t in range(1, T):
        model.addConstr(h_jt[j, t] >= H_j0[j])
        model.addConstr(h_jt[j, t] <= H_j1[j])
for i1 in range(I1):
    for t in range(1, T):
        for j in range(J):
            if j == jj[i1]:
                model.addConstr(l_i1jt[i1, j, t] * R_j[j] >= hv_i1_min[i1])
                model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_max[i1])
            else:
                model.addConstr(l_i1jt[i1, j, t] == 0)
for t in range(1, T):
    for j in range(J):
        if j == 2:
            model.addConstr(V_jt[j, t] == 0)
        else:
            model.addConstr(V_jt[0, t] - 9 * V_jt[1, t] == 0)

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
# 混合工序热值约束
for i2 in range(I2):
    for t in range(1, T):
        lhs_heat2 = gp.LinExpr()
        for j in range(J):
            L1 = L[i2]
            for l in range(L1):
                lhs_heat2 += R_j[j] * l_i2jt[i2, j, t, l]
        rhs_heat_min2 = hv_i2_min[i2]  # 最小热量值
        rhs_heat_max2 = hv_i2_max[i2]  # 最大热量值
        model.addConstr(lhs_heat2 >= rhs_heat_min2)
        model.addConstr(lhs_heat2 <= rhs_heat_max2)
# 电厂热值约束
for t in range(1, T):
    lhs_heat = gp.LinExpr()
    for j in range(J):
        lhs_heat.addTerms([R_j[j]], [V_jt[j, t]])
    model.addConstr(lhs_heat <= 50000 * 8243)
    # model.addConstr(lhs_heat >= 312450000)
z = {}
for t in range(1, T):
    for j in range(J):
        z[j, t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
for t in range(1, T):
    for j in range(J):
        model.addConstr(h_jt[j, t] - h_jt[j, t - 1] <= z[j, t])
        model.addConstr(h_jt[j, t - 1] - h_jt[j, t] <= z[j, t])
objective_expr = gp.LinExpr()
for t in range(1, T):
    hest = gp.LinExpr()
    for j in range(J):
        hest += V_jt[j, t] * R_j[j]
        objective_expr += c_jf[j] * F_jt[j, t]
        # objective_expr += (h_jt[j, t] - (H_j1[j] - H_j0[j]) / 2) * c_jf[j]
        objective_expr += z[j, t] * c_jf[j] * 0.1
    objective_expr += (50000 - hest / 8243) * 0.86
# 设置最小化目标
model.setParam(gp.GRB.Param.OutputFlag, 1)
model.Params.NonConvex = 2
model.setObjective(objective_expr, gp.GRB.MINIMIZE)
model.optimize()
data = []
for j in range(J):
    for t in range(1, T):
        data.append({"Variable": f"h_{j}_{t}", "Value": h_jt[j, t].x})
        data.append({"Variable": f"F_{j}_{t}", "Value": F_jt[j, t].x})
        for i1 in range(I1):
            data.append({"Variable": f"l_i1_{i1}_{j}_{t}", "Value": l_i1jt[i1, j, t].x})
        for i2 in range(I2):
            L1 = L[i2]
            for l in range(L1):
                data.append({"Variable": f"l_i2_{i2}_{j}_{t}_{l}", "Value": l_i2jt[i2, j, t, l].x})
for t in range(1, T):
    for i2 in range(I2):
        L1 = L[i2]
        for l in range(L1):
            data.append({"Variable": f"option_{i2}_{l}_{t}", "Value": e[i2, l, t].x})
for t in range(T):
    for j in range(J):
        # data.append({"Variable": f"z_{j}_{t}", "Value": z[j,t].x})
        data.append({"Variable": f"V_jt_{j}_{t}", "Value": V_jt[j, t].x})
df = pd.DataFrame(data)
# 保存 DataFrame 到 Excel 文件
output_file = "E:\新建文件夹\代码\lg\data\gp.xlsx"
df.to_excel(output_file, index=False)
print(f"Decision variables saved to {output_file}")
