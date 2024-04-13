import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import pandas as pd
df = pd.DataFrame(columns=["Variable", "Value"])
def generate_random_matrix(rows, cols, lower_limit, upper_limit):
    matrix = []
    for _ in range(rows):
        row_data = [random.randint(lower_limit, upper_limit) for _ in range(cols)]
        matrix.append(row_data)
    return matrix
row_count = 3
col_count = 24
# 第一行数据是1500到3000的随机数
matrix_1 = generate_random_matrix(1, col_count, 1500, 3000)
# 第二行是300到700的随机数
matrix_2 = generate_random_matrix(1, col_count, 300, 700)
# 第三行是30到200的随机数
matrix_3 = generate_random_matrix(1, col_count, 30, 200)
# 将三行矩阵合并成一个3行24列的矩阵
S_jt = np.vstack((matrix_1, matrix_2, matrix_3))
f_j = [1100,431,90]
model = gp.Model()
J = 3  # Number of J values
T = 24  # Number of T values
I1 = 6  # Number of I1 values
# I2 = 5  # Number of I2 values
O = {}
F_jt = {}
l_i1jt = {}
jj = [1,1,1,0,1,0]
for i1 in range(I1):
    for j in range(J):
        for t in range(T):
            l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
c_jf = [0.037,0.034,0.095]

for j in range(J):
    for t in range(T):
        O[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"O_{j}_{t}", lb=0)
        F_jt[j, t] = model.addVar(name=f"F_{j}_{t}")
for j in range(J):
    for t in range(T):
        model.addConstr(O[j, t] == f_j[j] * S_jt[j, t])
Celec = {t: 0.5 for t in range(T)}#t时段电价
#煤气柜上下线
H_j1 = {0:250000,1:150000,2:250000}
H_j0 = {0:50000,1:30000,2:50000}
hv_i1_min = [84771000,88320,178200,1430835.2,65280000,9050000]
hv_i1_max = [169542000,176640,243000,2861670.4,85280000,11050000]
R_j = [3350,18820,8364]
V_j =[1,1,1]
h_jt = {}
max_var = {}
seita = {}
condition = {}
z={}
for t in range(T):
    for j in range(J):
         condition[j,t]= model.addVar(vtype=GRB.BINARY,name=f"con{j}_{t}")
for j in range(J):
    for t in range(T):
        h_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{j}_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
        seita[j, t] = model.addVar(name=f"seita_{j}_{t}")
        z[j,t] = model.addVar(name=f"z_{j}_{t}")
        # V_j[j] = model.addVar(lb=0, name=f'V{j}')
initial_h_values = [150000,80000,150000]
for j in range(J):
    model.addConstr(h_jt[j, 0] == initial_h_values[j])
for t in range(1,T):
    for j in range(J):
        lhs = gp.LinExpr()
        lhs.addTerms([1], [O[j, t]])
        lhs.addTerms([V_j[j]], [h_jt[j, t - 1]])
        constant_expr = gp.LinExpr(-H_j0[j] * V_j[j])
        lhs.add(constant_expr)  # 将常数表达式添加到主线性表达式
        lhs.addTerms([-1], [seita[j, t]])
        for i in range(I1):
            lhs.addTerms([-1], [l_i1jt[i, j, t]])
        model.addConstr(lhs == 0)
condition1 = {}
condition2 = {}
M = 100000000000
for t in range(T):
    for j in range(J):
        # 添加条件判断约束

        # model.addConstr(seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j]*(1-condition[j,t])>=0)
        # model.addConstr(seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j]*condition[j,t]<=0)
        model.addConstr(z[j,t] >= (H_j1[j] - H_j0[j]) * V_j[j]-seita[j,t])
        # model.addConstr(z[j,t]*condition[j,t]>=-1)

        model.addConstr(h_jt[j,t] ==H_j1[j]*(1-condition[j,t])+(H_j0[j]+seita[j,t]/V_j[j])*condition[j,t])
        model.addConstr(F_jt[j,t] == (seita[j,t]-(H_j1[j] - H_j0[j]) * V_j[j])*(1-condition[j,t]))
for j in range(J):
    for t in range(T):
        model.addConstr(h_jt[j,t] >= H_j0[j])
        model.addConstr(h_jt[j,t] <= H_j1[j])
# for i1 in range(I1):
#     for t in range(T):
#         for j in range(J):
#           if j == jj[i1]:
#               model.addConstr(l_i1jt[i1, j, t] * R_j[j] >= hv_i1_min[i1])
#               model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_max[i1])
#           else:
#               model.addConstr(l_i1jt[i1, j, t] == 0)
for i1 in range(I1):
    for t in range(T):
        lhs_heat = gp.LinExpr()
        for j in range(J):
            lhs_heat.addTerms([R_j[j]], [l_i1jt[i1, j, t]])
        rhs_heat_min = hv_i1_min[i1]  # 最小热量值
        rhs_heat_max = hv_i1_max[i1]  # 最大热量值
        model.addConstr(lhs_heat >= rhs_heat_min)
        model.addConstr(lhs_heat <= rhs_heat_max)

objective_expr = gp.LinExpr()


for t in range(T):
    for j in range(J):
        objective_expr += F_jt[j, t]

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
for t in range(T):
    for j in range(J):
        data.append({"Variable": f"con{j}_{t}", "Value": condition[j,t].x})
        data.append({"Variable": f"z_{j}_{t}", "Value": z[j,t].x})
df = pd.DataFrame(data)
# 保存 DataFrame 到 Excel 文件
output_file = "demo3.xlsx"
df.to_excel(output_file, index=False)
print(f"Decision variables saved to {output_file}")

