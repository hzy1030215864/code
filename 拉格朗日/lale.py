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
# row_count = 18
# col_count = 24
# #烧结
# matrix_1 = generate_random_matrix(1, col_count, 1500, 3000)
# # 焦炉
# matrix_2 = generate_random_matrix(1, col_count, 400, 800)
# # 石灰
# matrix_3 = generate_random_matrix(1, col_count, 150, 200)
# # 第三行是30到200的随机数
# matrix_4 = generate_random_matrix(1, col_count, 1500, 1700)
# # 第三行是30到200的随机数
# matrix_5 = generate_random_matrix(1, col_count, 1400, 1600)
# # 第三行是30到200的随机数
# matrix_6 = generate_random_matrix(1, col_count, 840, 960)
# # 第三行是30到200的随机数
# matrix_7 = generate_random_matrix(1, col_count, 462, 528)
# # 第三行是30到200的随机数
# matrix_8 = generate_random_matrix(1, col_count, 98, 112)
# # 第三行是30到200的随机数
# matrix_9 = generate_random_matrix(1, col_count, 130, 140)
# # 第三行是30到200的随机数
# matrix_10 = generate_random_matrix(1, col_count, 1264, 1424)
# # 第三行是30到200的随机数
# matrix_11 = generate_random_matrix(1, col_count, 1264, 1424)
# # 第三行是30到200的随机数
# matrix_12 = generate_random_matrix(1, col_count, 1264, 1424)
# # 第三行是30到200的随机数
# matrix_13 = generate_random_matrix(1, col_count, 398.72, 455.68)
# # 第三行是30到200的随机数
# matrix_14 = generate_random_matrix(1, col_count, 145000, 145000)
# # 第三行是30到200的随机数
# matrix_15 = generate_random_matrix(1, col_count, 415, 415)
# # 第三行是30到200的随机数
# matrix_16 = generate_random_matrix(1, col_count, 110000, 110000)
# # 第三行是30到200的随机数
# matrix_17= generate_random_matrix(1, col_count, 4000, 4000)
# # 第三行是30到200的随机数
# matrix_18= generate_random_matrix(1, col_count, 3000, 3000)
# # 将三行矩阵合并成一个3行24列的矩阵
# S_it = np.vstack((matrix_1,matrix_2,matrix_3,matrix_4,matrix_5,matrix_6,matrix_7,matrix_8,matrix_9,matrix_10,matrix_11,matrix_12,matrix_13,matrix_14,matrix_15,matrix_16,matrix_17,matrix_18))
# S_jt = np.vstack((matrix_4, matrix_2, matrix_5))
# # 将矩阵转换为DataFrame
# df = pd.DataFrame(S_jt)
# # 将DataFrame保存到Excel文件
# excel_file = 'S_jt.xlsx'
# df.to_excel(excel_file, index=False)
# df = pd.DataFrame(S_it)
# excel_file1 = "S_it.xlsx"
# df.to_excel(excel_file1, index=False)
loaded_df = pd.read_excel("S_jt.xlsx")
# 将数据转换回矩阵（NumPy数组）
S_jt = loaded_df.to_numpy()
loaded_df1 = pd.read_excel("S_it.xlsx")
# 将数据转换回矩阵（NumPy数组）
S_it = loaded_df1.to_numpy()

f_j = [1100,431,90]
model = gp.Model()
J = 3  # Number of J values
T = 24  # Number of T values
I1 = 7  # Number of I1 values
I2 = 11  # Number of I2 values
L=[3,1,3,1,1,2,2,2,1,4,1]
O = {}
F_jt = {}
l_i1jt = {}
l_i2jt = {}
erfa_j = [0.35,0.35,0.35]
jj = [1,1,1,1,0,1,0]
erlfa11 = {}
erlfa22 = {}
erlfa33={}
max_iterations = 10  # 最大迭代次数
lambda_i1jt1 = {}
lambda_i1jt2 = {}
lambda_i1jt3 = {}
sete = {}
l2_value = {}
e_value = {}
for i in range(I2):
    for t in range(1,T):
        lambda_i1jt1[i,t]= 0
for i in range(I2):
    for t in range(1,T):
        lambda_i1jt2[i,t]= 0
for i in range(I2):
    for t in range(1,T):
        lambda_i1jt3[i,t]= 0
for iteration in range(max_iterations):
    for i1 in range(I1):
        for j in range(J):
            for t in range(T):
                l_i1jt[i1, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i1_{i1}_{j}_{t}", lb=0)
    for i2 in range(I2):
        for j in range(J):
            for t in range(T):
                l_i2jt[i2, j, t] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"l_i2_{i2}_{j}_{t}", lb=0)
    c_jf = [0.037, 0.343, 0.095]
    V_jt = {}
    for j in range(J):
        for t in range(T):
            O[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"O_{j}_{t}", lb=0)
            F_jt[j, t] = model.addVar(name=f"F_{j}_{t}")
            V_jt[j, t] = model.addVar(name=f"V_jt_{j}_{t}")
    for j in range(J):
        for t in range(T):
            model.addConstr(O[j, t] == f_j[j] * S_jt[j, t])
    Celec = {t: 0.5 for t in range(T)}  # t时段电价
    # 煤气柜上下线
    H_j1 = {0: 240000, 1: 100000, 2: 207000}
    H_j0 = {0: 140000, 1: 65000, 2: 50000}
    danyi = [0, 4, 5, 8, 13, 16, 17]
    hunhe = [1, 2, 3, 6, 7, 9, 10, 11, 12, 14, 15]
    hv_i1_min = {}
    hv_i2_min = {}
    # hv_i1_min = [84771000,88155200,309120,210600,1154020000,65870000,10050000]
    # hv_i1_max = [169542000,100748800,353280,226800,1236450000,84690000,10050000]
    # hv_i2_min = [1469252000,8472000,2988180000,	5008080,3502520,16696400,19175940,19175940,5007923.2,879740000,836000000]
    # hv_i2_max = [2938504000,11296000,3386604000,5723520,4002880,19081600,21915360,21915360,5723340.8,945720500,1003200000]
    w = [56514, 3673130, 56480, 1992120, 62968, 368, 10840, 35740, 1620, 13400, 15390, 15390, 12560, 8243, 2199350,
         8360, 18820, 3350]
    for i1 in range(I1):
        i = danyi[i1]
        for t in range(T):
            hv_i1_min[i1, t] = w[i] * S_it[i, t]
    for i2 in range(I2):
        i = hunhe[i2]
        for t in range(T):
            hv_i2_min[i2, t] = w[i] * S_it[i, t]
    R_j = [3350, 18820, 8364]
    # hve_min = 8243000000
    hve_min = 0
    hve_max = 9067300000
    h_jt = {}
    max_var = {}
    hve = {}
    e = {}
    erlfa1 = {}
    erlfa2 = {}
    erlfa3 = {}
    erlfa = {}
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
    for j in range(J):
        for t in range(T):
            h_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{j}_{t}", lb=0)
            max_var[t] = model.addVar(lb=0, name=f'max_var_{t}')
            hve[j, t] = model.addVar(name=f"hve_{j}_{t}")
    initial_h_values = [200000, 80000, 150000]
    for j in range(J):
        model.addConstr(h_jt[j, 0] == initial_h_values[j])
    for j in range(J):
        for t in range(1, T):
            lhs1 = gp.LinExpr()
            lhs1.addTerms([1], [O[j, t]])
            lhs1.addTerms([1], [h_jt[j, t - 1]])
            for i1 in range(I1):
                lhs1.addTerms([-1], [l_i1jt[i1, j, t]])
            for i2 in range(I2):
                lhs1.addTerms([-1], [l_i2jt[i2, j, t]])
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
                    model.addConstr(l_i1jt[i1, j, t] * R_j[j] == hv_i1_min[i1, t])
                    # model.addConstr(l_i1jt[i1, j, t] * R_j[j] <= hv_i1_min[i1,t])
                else:
                    model.addConstr(l_i1jt[i1, j, t] == 0)
    for t in range(1, T):
        for i in range(I2):
            L1 = L[i]
            for l in range(L1):
                e[i, l, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"option_{i}_{l}_{t}")
    # 混合比例约束
    # for i in range(I2):
    #     for t in range(1,T):
    #         L1 = L[i]
    #         lll = gp.QuadExpr()
    #         lll1 = gp.QuadExpr()
    #         lll2 = gp.QuadExpr()
    #         for l in range(L1):
    #             value1 = erf[i][l][0]
    #             value2 = erf[i][l][1]
    #             value3 = erf[i][l][2]
    #             lll.addTerms(value2,  e[i, l, t],l_i2jt[i, 0, t])
    #             lll.addTerms(-value1, e[i, l, t], l_i2jt[i, 1, t])
    #             lll1.addTerms(value3,  e[i, l, t],l_i2jt[i, 0, t])
    #             lll1.addTerms(-value1, e[i, l, t], l_i2jt[i, 2, t])
    #             lll2.addTerms(value3,  e[i, l, t],l_i2jt[i, 1, t])
    #             lll2.addTerms(-value2, e[i, l, t], l_i2jt[i, 2, t])
    #             # model.addConstr(lll == 0)
    #             # model.addConstr(lll1 == 0)
    #             # model.addConstr(lll2 == 0)
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
                lhs_heat2 += R_j[j] * l_i2jt[i2, j, t]
            rhs_heat_min2 = hv_i2_min[i2, t]  # 最小热量值
            rhs_heat_max2 = hv_i2_min[i2, t]  # 最大热量值
            model.addConstr(lhs_heat2 == rhs_heat_min2)
            # model.addConstr(lhs_heat2 <= rhs_heat_max2)
    objective_expr = gp.LinExpr()
    for t in range(1, T):
        hest = gp.LinExpr()
        for j in range(J):
            hest += V_jt[j, t] * R_j[j]
            objective_expr += c_jf[j] * F_jt[j, t]
            # objective_expr += (h_jt[j, t] - (H_j1[j] - H_j0[j]) / 2) * c_jf[j]
            objective_expr += (h_jt[j, t] - h_jt[j, t - 1]) * c_jf[j]
        objective_expr += (1050000 - hest / 3600) * 0.86
    # 设置最小化目标
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
                lll.addTerms(value2, e[i, l, t], l_i2jt[i, 0, t])
                lll.addTerms(-value1, e[i, l, t], l_i2jt[i, 1, t])
                lll1.addTerms(value3, e[i, l, t], l_i2jt[i, 0, t])
                lll1.addTerms(-value1, e[i, l, t], l_i2jt[i, 2, t])
                lll2.addTerms(value3, e[i, l, t], l_i2jt[i, 1, t])
                lll2.addTerms(-value2, e[i, l, t], l_i2jt[i, 2, t])
            objective_expr +=lambda_i1jt1[i,t]*(lll)
            objective_expr += lambda_i1jt2[i, t] * (lll1)
            objective_expr += lambda_i1jt3[i, t] * (lll2)
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    model.Params.NonConvex = 2
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    model.optimize()
    # l1_total = 0
    l2_total = 0
    for t in range(1, T):
        for i2 in range(I2):
            for j in range(J):
                l2_var = l_i2jt[i2, j, t]
                l2_value[i2,j,t] = l2_var.x
                # l2_total += l2_value
            L1 = L[i2]
            for l in range(L1):
                e_var = e[i2,l,t]
                e_value[i2,l,t] = e_var.x

    for i in range(I2):
        for t in range(1,T):
            llll = 0
            llll1 = 0
            llll2 = 0
            L1 = L[i]
            for l in range(L1):
                value1 = erf[i][l][0]
                value2 = erf[i][l][1]
                value3 = erf[i][l][2]
                llll += value2*e_value[i, l, t]*l2_value[i, 0, t]-value1*e_value[i, l, t]*l2_value[i, 1, t]
                llll1 += value3*e_value[i, l, t]*l2_value[i, 0, t]-value1*e_value[i, l, t]*l2_value[i, 2, t]
                llll2 +=value3*e_value[i, l, t]*l2_value[i, 1, t]-value2*e_value[i, l, t]*l2_value[i, 2, t]
            erlfa1[i,t] = llll
            erlfa2[i,t] = llll1
            erlfa3[i,t] = llll2
            # erlfa11[i,t] = np.linalg.norm(erlfa1[i,t])
            # erlfa22[i,t] = np.linalg.norm(erlfa2[i,t])
            # erlfa33[i,t] = np.linalg.norm(erlfa3[i,t])
    for i in range(I2):
        for t in range(1,T):
            if erlfa1[i,t] == 0:
                lambda_i1jt1[i,t] =lambda_i1jt1[i,t]
            else:
                objective_value = objective_expr.getValue()
                sete[i, t] = (3160000 - objective_value) / (erlfa1[i,t])**2
                fm = erlfa1[i,t]
                lambda_i1jt1[i, t] =lambda_i1jt1[i,t]+ sete[i, t]
            if erlfa2[i,t]  == 0:
                lambda_i1jt2[i,t] =lambda_i1jt2[i,t]
            else:
                objective_value = objective_expr.getValue()
                sete[i, t] = (3160000 - objective_value) / (erlfa2[i,t])**2
                fm = erlfa2[i,t]
                lambda_i1jt2[i, t] += sete[i, t]
            if erlfa3[i,t]  == 0:
                lambda_i1jt3[i,t] =lambda_i1jt3[i,t]
            else:
                objective_value = objective_expr.getValue()
                sete[i, t] = (3160000 - objective_value) / (erlfa3[i,t])**2
                fm =erlfa3[i,t]
                lambda_i1jt3[i, t] += sete[i, t]

    print(iteration)
    print(lambda_i1jt1)
    # print(l_i1jt)
    # print(l_i2jt)

# data = []
# for j in range(J):
#         for t in range(1,T):
#             data.append({"Variable": f"h_{j}_{t}", "Value": h_jt[j, t].x})
#             data.append({"Variable": f"F_{j}_{t}", "Value": F_jt[j, t].x})
#             for i1 in range(I1):
#                 data.append({"Variable": f"l_i1_{i1}_{j}_{t}", "Value": l_i1jt[i1, j, t].x})
#             for i2 in range(I2):
#                 data.append({"Variable": f"l_i2_{i2}_{j}_{t}", "Value": l_i2jt[i2, j, t].x})
# for t in range(1, T):
#     for i2 in range(I2):
#             L1 = L[i2]
#             for l in range(L1):
#                 data.append({"Variable": f"option_{i2}_{l}_{t}", "Value": e[i2,l,t].x})
# for t in range(T):
#     for j in range(J):
#         # data.append({"Variable": f"z_{j}_{t}", "Value": z[j,t].x})
#         data.append({"Variable": f"V_jt_{j}_{t}", "Value":  V_jt[j,t].x})
# df = pd.DataFrame(data)
# # 保存 DataFrame 到 Excel 文件
# output_file = "lale.xlsx"
# df.to_excel(output_file, index=False)
# print(f"Decision variables saved to {output_file}")

