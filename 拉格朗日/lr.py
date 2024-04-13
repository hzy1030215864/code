import gurobipy as gp

# 创建一个Gurobi模型
model = gp.Model()

# 添加变量到模型
x = {}
x[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0)
x[1] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0)

# 更新模型以反映变量的添加
model.update()

# 设置最大迭代次数
max_iterations = 200
lamda = 0
# 创建 lambda_i1jt1 和 x_vale 列表并初始化

x_vale = [(0, 0)]
convergence_threshold = 1e-4
# 迭代过程
for iteration in range(max_iterations):
    objective_expr = gp.LinExpr()
    objective_expr = x[0] + 2*x[1]+lamda*(2-x[0] - x[1])
    # 添加约束
    model.addConstr(x[0] >= 0)
    model.addConstr(x[1] >= 0)

    # 设置模型的目标函数
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)
    # 求解模型
    model.optimize()
    model.update()
    x_vale.append((x[0].x, x[1].x))



    # 获取变量值并添加到 x_vale 列表中
    # 计算 setpt
    numerator = 1 - objective_expr.getValue()
    denominator = 2 - x_vale[iteration + 1][0] - x_vale[iteration + 1][1]
    numerator_squared = numerator
    denominator_squared = denominator * denominator
    setpt = numerator_squared / denominator_squared
    lamda = lamda+0.5*setpt*denominator
    # 更新 lambda_i1jt1

    print(f"Iteration: {iteration}, x: {x_vale}")
    # if iteration > 0:
    #     objective_change = abs(2 - objective_expr.getValue())
    #
    #     # 如果目标函数值的变化小于阈值，停止迭代
    #     if objective_change < convergence_threshold:
    #         print(f"Converged after {iteration + 1} iterations.")
    #         break
