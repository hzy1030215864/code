import pandas as pd
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取两个 Excel 文件
df_gru = pd.read_excel('GRUCOG发生.xlsx')
df_tcn_gru = pd.read_excel('TCN+GRUCOG发生.xlsx')
print(df_tcn_gru)
print(df_gru)
# 绘制对比图表
plt.plot(df_gru['真实值'], label='真实值', marker='o')
plt.plot(df_gru['预测值'], label='GRU预测', marker='o')
plt.plot(df_tcn_gru['预测值'], label='TCN+GRU预测', marker='o')
plt.legend()
plt.show()

# 保存结果图表
plt.savefig('prediction_comparison.png')
