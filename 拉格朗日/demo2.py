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
l_i1jt = {}
for i1 in range(I1):
    for j in range(J):
        for t in range(T):
            l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
c_jf = [0.037,0.034,0.095]

for j in range(J):
    for t in range(T):
        O[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"O_{j}_{t}", lb=0)
for j in range(J):
    for t in range(T):
        model.addConstr(O[j, t] == f_j[j] * S_jt[j, t])
Celec = {t: 0.5 for t in range(T)}#t时段电价
#煤气柜上下线
H_j1 = {0:300000,1:150000,2:250000}
H_j0 = {0:100000,1:50000,2:100000}
hv_i1_min = [84771000,88320,178200,1430835.2,65280000,9050000]
hv_i1_max = [169542000,176640,243000,2861670.4,85280000,11050000]
R_j = [3350,18820,8364]
V_j =[10,20,15]
h_jt = {}
max_var = {}
for j in range(J):
    for t in range(T):
        h_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{j}_{t}", lb=0)
        max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
initial_h_values = [200000,80000,150000]
for j in range(J):
    model.addConstr(h_jt[j, 0] == initial_h_values[j])

#煤气平衡约束
objective_expr = gp.LinExpr()
for t in range(1,T):
    for j in range(J):
        lhs = gp.LinExpr()
        lhs.addTerms([V_j[j]], [h_jt[j, t - 1]])
        lhs.addTerms([1], [O[j, t]])

        # 添加约束右侧的各项
        rhs = gp.LinExpr()
        for i in range(I1):
            rhs.addTerms([1], [l_i1jt[i, j, t]])

        rhs.addTerms([V_j[j]], [h_jt[j,t]])
        # x = model.addVar(name="x")
        # aux = model.addVar(vtype=gp.GRB.CONTINUOUS, name="aux")  # 辅助变量

        # # 添加约束
        # model.addConstr(aux >= 0)
        # model.addConstr(aux >= x)
        # model.addConstr(aux <= x)  # 当 x >= 0 时，aux = x；当 x < 0 时，aux = 0

        # 添加 max 操作

        model.addConstr(max_var[t]>=0)
        model.addConstr(max_var[t] >= O[j, t] + h_jt[j, t - 1] * V_j[j] -
                        gp.quicksum(l_i1jt[i, j, t] for i in range(I1))
                        - H_j1[j] *V_j[j])
        # model.addConstr(max_var[t] <= O[j, t] + h_jt[j, t - 1] * V_j[j] -
        #                 gp.quicksum(l_i1jt[i, j, t] for i in range(I1))
        #                 - H_j1[j] * V_j[j])
        rhs.addTerms([1], [max_var[t]])

        # 添加约束 h_(j,t-1)∙V_j + O_jt = ∑_(i_1)^(I_1)▒l_(i_1 jt) + h_(j,t)∙V_j + max...
        model.addConstr(lhs, gp.GRB.EQUAL, rhs, name=f'equation_{j}_{t}')
        objective_expr.addTerms([c_jf[j]], [max_var[t]])
for j in range(J):
    for t in range(T):
        model.addConstr(h_jt[j,t] >= H_j0[j])
        model.addConstr(h_jt[j, t] <= H_j1[j])
for i1 in range(I1):
    for t in range(T):
        for j in range(J):
            model.addConstr(l_i1jt[i1, j, t] * R_j[j] >= hv_i1_min[i1])
            model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_max[i1])

F_1 = model.addVar(name='F_1')

model.setObjective(F_1 + objective_expr, gp.GRB.MINIMIZE)

# 优化模型
model.optimize()
data = []
for j in range(J):
    for i1 in range(I1):
        for t in range(T):
            data.append({"Variable": f"h_{j}_{t}", "Value": h_jt[j, t].x})
            data.append({"Variable": f"l_i1_{i1}_{j}_{t}", "Value": l_i1jt[i1, j, t].x})
            data.append({"Variable": f"max_var_{t}", "Value": max_var[t].x})
df = pd.DataFrame(data)
# 保存 DataFrame 到 Excel 文件
output_file = "decision_variables.xlsx"
df.to_excel(output_file, index=False)
print(f"Decision variables saved to {output_file}")