import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import pandas as pd
df = pd.DataFrame(columns=["Variable", "Value"])
# def generate_random_matrix(rows, cols, lower_limit, upper_limit):
#     matrix = []
#     for _ in range(rows):
#         row_data = [random.randint(lower_limit, upper_limit) for _ in range(cols)]
#         matrix.append(row_data)
#     return matrix
# row_count = 3
# col_count = 24
# # 第一行数据是1500到3000的随机数
# matrix_1 = generate_random_matrix(1, col_count, 1500, 1800)
# # 第二行是300到700的随机数
# matrix_2 = generate_random_matrix(1, col_count, 400, 800)
# # 第三行是30到200的随机数
# matrix_3 = generate_random_matrix(1, col_count, 1500, 1800)
# # 将三行矩阵合并成一个3行24列的矩阵
# S_jt = np.vstack((matrix_1, matrix_2, matrix_3))
loaded_df = pd.read_excel("S_jt.xlsx")
# 将数据转换回矩阵（NumPy数组）
S_jt = loaded_df.to_numpy()
f_j = [1100,431,90]
model = gp.Model()
J = 3  # Number of J values
T = 24  # Number of T values
I1 = 10  # Number of I1 values
I2 = 4  # Number of I2 values
L=10
O = {}
F_jt = {}
l_i1jt = {}
l_i2jt = {}
erfa_j = [0.35,0.35,0.35]
jj = [1,1,1,1,1,1,1,0,1,0]
for i1 in range(I1):
    for j in range(J):
        for t in range(T):
            l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
for i2 in range(I2):
    for j in range(J):
        for t in range(T):
            l_i2jt[i2, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{j}_{t}", lb=0)
c_jf = [0.037,0.034,0.095]
V_jt = {}
for j in range(J):
    for t in range(T):
        O[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"O_{j}_{t}", lb=0)
        F_jt[j, t] = model.addVar(name=f"F_{j}_{t}")
        V_jt[j,t]=model.addVar(name=f"V_jt_{j}_{t}")
for j in range(J):
    for t in range(T):
        model.addConstr(O[j, t] == f_j[j] * S_jt[j, t])
Celec = {t: 0.5 for t in range(T)}#t时段电价
#煤气柜上下线
H_j1 = {0:250000,1:150000,2:250000}
H_j0 = {0:50000,1:30000,2:50000}
hv_i1_min = [84771000,88320,1430880,1000720,178200,1430835.2,65280000,9050000,7000000,1100000000]
hv_i1_max = [169542000,176640,2861760,2001440,243000,2861670.4,85280000,11050000,8000000,1300000000]
hv_i2_min = [4770400,5478840,5478840,600000000]
hv_i2_max = [9540800,10957680,10957680,1100000000]

R_j = [3350,18820,8364]
hve_min = 1000000000
hve_max = 4000000000
V_j =[1,1,1]
h_jt = {}
max_var = {}
seita = {}
condition = {}
z={}
hve = {}
for t in range(T):
    for j in range(J):
         condition[j,t]= model.addVar(vtype=GRB.BINARY,name=f"con{j}_{t}")
for j in range(J):
    for t in range(T):
        h_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{j}_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
        seita[j, t] = model.addVar(name=f"seita_{j}_{t}")
        z[j,t] = model.addVar(name=f"z_{j}_{t}")
        hve[j,t] = model.addVar(name=f"hve_{j}_{t}")
        # V_j[j] = model.addVar(lb=0, name=f'V{j}')
initial_h_values = [150000,80000,150000]
for j in range(J):
    model.addConstr(h_jt[j, 0] == initial_h_values[j])
# for t in range(1,T):
#     for j in range(J):
#         lhs = gp.LinExpr()
#         lhs.addTerms([1], [O[j, t]])
#         lhs.addTerms([V_j[j]], [h_jt[j, t - 1]])
#         constant_expr = gp.LinExpr(-H_j0[j] * V_j[j])
#         lhs.add(constant_expr)  # 将常数表达式添加到主线性表达式
#         lhs.addTerms([-1], [seita[j, t]])
#         lhs.addTerms([-1], [V_jt[j,t]])
#         for i1 in range(I1):
#             lhs.addTerms([-1], [l_i1jt[i1, j, t]])
#         for i2 in range(I2):
#             lhs.addTerms([-1], [l_i2jt[i2, j, t]])
#         model.addConstr(lhs == 0)
# condition1 = {}
# condition2 = {}
# M = 100000000000
# for t in range(T):
#     for j in range(J):
#         # 添加条件判断约束
#
#         # model.addConstr(seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j]*(1-condition[j,t])>=0)
#         # model.addConstr(seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j]*condition[j,t]<=0)
#         # model.addConstr(z[j,t] >= (H_j1[j] - H_j0[j]) * V_j[j]-seita[j,t])
#         # model.addConstr(z[j,t]*condition[j,t]>=-1)
#
#         model.addConstr(h_jt[j,t] ==H_j1[j]*(1-condition[j,t])+(H_j0[j]+seita[j,t]/V_j[j])*condition[j,t])
#         model.addConstr(F_jt[j,t] == (seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j])*(1-condition[j,t]))
for j in range(J):
    for t in range(1,T):
        lhs1 = gp.LinExpr()
        lhs1.addTerms([1], [O[j, t]])
        lhs1.addTerms([V_j[j]], [h_jt[j, t - 1]])
        for i1 in range(I1):
            lhs1.addTerms([-1], [l_i1jt[i1, j, t]])
        for i2 in range(I2):
            lhs1.addTerms([-1], [l_i2jt[i2, j, t]])
        lhs1.addTerms([-1], [h_jt[j, t]])
        lhs1.addTerms([-1], [V_jt[j, t]])
        lhs1.addTerms([-1], [F_jt[j, t]])
        model.addConstr(lhs1 == 0)
for j in range(J):
    for t in range(T):
        model.addConstr(h_jt[j,t] >= H_j0[j])
        model.addConstr(h_jt[j,t] <= H_j1[j])
for i1 in range(I1):
    for t in range(T):
        for j in range(J):
          if j == jj[i1]:
              model.addConstr(l_i1jt[i1, j, t] * R_j[j] >= hv_i1_min[i1])
              model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_max[i1])
          else:
              model.addConstr(l_i1jt[i1, j, t] == 0)
for i1 in range(I1):
    for t in range(T):
        lhs_heat1 = gp.LinExpr()
        for j in range(J):
            lhs_heat1.addTerms([R_j[j]], [l_i1jt[i1, j, t]])
        rhs_heat_min1 = hv_i1_min[i1]  # 最小热量值
        rhs_heat_max1 = hv_i1_max[i1]  # 最大热量值
        model.addConstr(lhs_heat1 >= rhs_heat_min1)
        model.addConstr(lhs_heat1 <= rhs_heat_max1)
for t in range(T):
    for i1 in range(I1):
        lhs = gp.LinExpr()
        for j in range(J):
            if j != jj[i1]:
                lhs.addTerms([1], [l_i1jt[i1, j, t]])
        model.addConstr(lhs == 0)  # 保证工序 i1 只使用煤气 j1
e = {}
erf = {}
num_options = 3
for t in range(T):
    for i in range(I2):
        for j in range(J):
            e[i, j, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{j}_{t}")
            # erf[i, l, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"gas_mixing_option_{i}_{l}_{t}")
for i in range(I2):
    for t in range(T):
         model.addConstr(gp.quicksum(e[i, j, t] for j in range(J)) == 1, f"process_{i}_constraint")
for t in range(T):
    # Add constraints for gas scheme selection in the first process
    model.addConstr(e[0, 1, t] + e[0, 2, t] == 1)
    model.addConstr(e[1, 1, t] + e[1, 2, t] == 1)
    model.addConstr(e[2, 1, t] + e[2, 2, t] == 1)
    model.addConstr(e[3, 1, t] + e[3, 2, t] == 1)
for t in range(T):
    model.addConstr(e[0, 1, t]*R_j[1]*l_i2jt[0,1,t]+e[0, 2, t]*R_j[2]*l_i2jt[0,2,t]>=hv_i2_min[0])
    model.addConstr(e[0, 1, t]*R_j[1]*l_i2jt[0,1,t]+e[0, 2, t]*R_j[2]*l_i2jt[0,2,t]<=hv_i2_max[0])
    model.addConstr(l_i2jt[0,0,t] == 0)
    model.addConstr(l_i2jt[1, 0, t] == 0)
    model.addConstr(l_i2jt[2, 0, t] == 0)
    model.addConstr(l_i2jt[3, 0, t] == 0)
    model.addConstr(e[1, 1, t]*R_j[1]*l_i2jt[1,1,t]+e[1, 2, t]*R_j[2]*l_i2jt[1,2,t]>=hv_i2_min[1])
    model.addConstr(e[1, 1, t]*R_j[1]*l_i2jt[1,1,t]+e[1, 2, t]*R_j[2]*l_i2jt[1,2,t]<=hv_i2_max[1])
    model.addConstr(e[2, 1, t]*R_j[1]*l_i2jt[2,1,t]+e[2, 2, t]*R_j[2]*l_i2jt[2,2,t]>=hv_i2_min[2])
    model.addConstr(e[2, 1, t]*R_j[1]*l_i2jt[2,1,t]+e[2, 2, t]*R_j[2]*l_i2jt[2,2,t]<=hv_i2_max[2])
    model.addConstr(e[3, 1, t]*R_j[1]*l_i2jt[3,1,t]+e[3, 2, t]*R_j[2]*l_i2jt[3,2,t]>=hv_i2_min[3])
    model.addConstr(e[3, 1, t]*R_j[1]*l_i2jt[3,1,t]+e[3, 2, t]*R_j[2]*l_i2jt[3,2,t]<=hv_i2_max[3])
for i in range(I2):
    for j in range(J):
        for t in range(T):
            model.addConstr(l_i2jt[i,j,t] <= e[i, j,t] * 10000000)  # 使用M大数法
for i2 in range(I2):
    for t in range(T):
        lhs_heat2 = gp.LinExpr()
        for j in range(J):
            coefficient = R_j[j] * l_i2jt[i2, j, t]
            lhs_heat2 += coefficient * e[i2, j, t]

        rhs_heat_min2 = hv_i2_min[i2]  # 最小热量值
        rhs_heat_max2 = hv_i2_max[i2]  # 最大热量值

        # 构建约束表达式
        # 添加约束到模型
        model.addConstr(lhs_heat2 >= rhs_heat_min2)
        model.addConstr(lhs_heat2 <= rhs_heat_max2)

for j in range(J):
    for t in range(T):
        lhs_heat = gp.LinExpr()
        lhs_heat.addTerms([R_j[j]],[V_jt[j,t]])
        model.addConstr(lhs_heat <= hve_max)
        model.addConstr(lhs_heat >= hve_min)


objective_expr = gp.LinExpr()


for t in range(1,T):
    for j in range(J):
        objective_expr += c_jf[j]*F_jt[j, t]
        objective_expr -=erfa_j[j]*V_jt[j,t]*(Celec[t]-0.3)
        objective_expr +=(h_jt[j,t]-(H_j1[j]-H_j0[j])/2)*0.01

# 设置最小化目标
model.setParam(gp.GRB.Param.OutputFlag, 1)
model.setObjective(objective_expr, gp.GRB.MINIMIZE)
model.optimize()
data = []
for j in range(J):
        for t in range(T):
            data.append({"Variable": f"h_{j}_{t}", "Value": h_jt[j, t].x})
            data.append({"Variable": f"F_{j}_{t}", "Value": F_jt[j, t].x})
            data.append({"Variable": f"seita_{j}_{t}", "Value": seita[j, t].x})
            for i1 in range(I1):
                data.append({"Variable": f"l_i1_{i1}_{j}_{t}", "Value": l_i1jt[i1, j, t].x})
            for i2 in range(I2):
                data.append({"Variable": f"l_i2_{i2}_{j}_{t}", "Value": l_i2jt[i2, j, t].x})
for t in range(T):
    for j in range(J):
        data.append({"Variable": f"con{j}_{t}", "Value": condition[j,t].x})
        # data.append({"Variable": f"z_{j}_{t}", "Value": z[j,t].x})
        data.append({"Variable": f"V_jt_{j}_{t}", "Value":  V_jt[j,t].x})
df = pd.DataFrame(data)
# 保存 DataFrame 到 Excel 文件
output_file = "demo6.xlsx"
df.to_excel(output_file, index=False)
print(f"Decision variables saved to {output_file}")

