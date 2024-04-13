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
row_count = 18
col_count = 480
#烧结
matrix_1 = generate_random_matrix(1, col_count, 1500, 3000)
# 焦炉
matrix_2 = generate_random_matrix(1, col_count, 400, 800)
# 石灰
matrix_3 = generate_random_matrix(1, col_count, 150, 200)
# 第三行是30到200的随机数
matrix_4 = generate_random_matrix(1, col_count, 1500, 1700)
# 第三行是30到200的随机数
matrix_5 = generate_random_matrix(1, col_count, 1400, 1600)
# 第三行是30到200的随机数
matrix_6 = generate_random_matrix(1, col_count, 840, 960)
# 第三行是30到200的随机数
matrix_7 = generate_random_matrix(1, col_count, 462, 528)
# 第三行是30到200的随机数
matrix_8 = generate_random_matrix(1, col_count, 98, 112)
# 第三行是30到200的随机数
matrix_9 = generate_random_matrix(1, col_count, 130, 140)
# 第三行是30到200的随机数
matrix_10 = generate_random_matrix(1, col_count, 1264, 1424)
# 第三行是30到200的随机数
matrix_11 = generate_random_matrix(1, col_count, 1264, 1424)
# 第三行是30到200的随机数
matrix_12 = generate_random_matrix(1, col_count, 1264, 1424)
# 第三行是30到200的随机数
matrix_13 = generate_random_matrix(1, col_count, 398, 455)
# 第三行是30到200的随机数
matrix_14 = generate_random_matrix(1, col_count, 145000, 145000)
# 第三行是30到200的随机数
matrix_15 = generate_random_matrix(1, col_count, 415, 415)
# 第三行是30到200的随机数
matrix_16 = generate_random_matrix(1, col_count, 110000, 110000)
# 第三行是30到200的随机数
matrix_17= generate_random_matrix(1, col_count, 4000, 4000)
# 第三行是30到200的随机数
matrix_18= generate_random_matrix(1, col_count, 3000, 3000)
# 将三行矩阵合并成一个3行24列的矩阵
S_it = np.vstack((matrix_1,matrix_2,matrix_3,matrix_4,matrix_5,matrix_6,matrix_7,matrix_8,matrix_9,matrix_10,matrix_11,matrix_12,matrix_13,matrix_14,matrix_15,matrix_16,matrix_17,matrix_18))
S_jt = np.vstack((matrix_4, matrix_2, matrix_5))
# 将矩阵转换为DataFrame
df = pd.DataFrame(S_jt)
# 将DataFrame保存到Excel文件
excel_file = 'D:\Python38\lg\data\S_jt_24.xlsx'
df.to_excel(excel_file, index=False)
df = pd.DataFrame(S_it)
excel_file1 = "D:\Python38\lg\data\S_it_24.xlsx"
df.to_excel(excel_file1, index=False)