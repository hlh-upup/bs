import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# -------------------------- 全局配置（解决中文显示、图表样式）--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文支持（SimHei适配Windows）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
rcParams['figure.figsize'] = (15, 6)  # 图表总尺寸
rcParams['axes.linewidth'] = 0.8  # 坐标轴线条宽度
rcParams['font.size'] = 10  # 基础字体大小
rcParams['legend.fontsize'] = 9  # 图例字体大小

# -------------------------- 数据整理 --------------------------
# 1. 皮尔逊相关系数（PCC）数据
pcc_metrics = ['总体感知质量', '音频质量', '跨模态一致性', '唇形同步度', '表情自然度']
pcc_values = [0.698, 0.621, 0.596, 0.554, 0.437]
pcc_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']  # 渐变配色

# 2. 预测误差分布数据
error_metrics = ['总体评分', '跨模态一致性', '表情自然度']
me_values = [-0.1434, -0.1316, -0.1439]  # 平均误差
sd_values = [0.3753, 0.3745, 0.3867]     # 标准差
error_x = np.arange(len(error_metrics))  # x轴位置
width = 0.35  # 柱状图宽度

# -------------------------- 创建子图 --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.2]})

# -------------------------- 子图1：皮尔逊相关系数柱状图 --------------------------
bars1 = ax1.bar(pcc_metrics, pcc_values, color=pcc_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax1.set_ylim(0, 0.8)  # y轴范围（突出差异）
ax1.set_ylabel('皮尔逊相关系数 (PCC)', fontsize=11, fontweight='bold')
ax1.set_title('各质量子指标与真实值的线性相关性', fontsize=12, fontweight='bold', pad=20)

# 在柱子顶部添加数值标签（保留3位小数）
for bar, value in zip(bars1, pcc_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 添加网格线（增强可读性）
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)  # 网格线置于底层

# -------------------------- 子图2：预测误差分布统计图（双柱状图） --------------------------
bars2_me = ax2.bar(error_x - width/2, me_values, width, label='平均误差 (ME)', 
                   color='#3498DB', alpha=0.8, edgecolor='white', linewidth=1.5)
bars2_sd = ax2.bar(error_x + width/2, sd_values, width, label='标准差 (SD)', 
                   color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=1.5)

# 设置y轴范围（适配ME和SD的数值差异）
ax2.set_ylim(-0.2, 0.5)
ax2.set_ylabel('误差值', fontsize=11, fontweight='bold')
ax2.set_title('各质量子指标的预测误差分布', fontsize=12, fontweight='bold', pad=20)
ax2.set_xticks(error_x)
ax2.set_xticklabels(error_metrics)
ax2.legend(loc='upper right', framealpha=0.9)

# 在柱子顶部添加数值标签（ME保留4位小数，SD保留4位小数）
for bar, value in zip(bars2_me, me_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height - 0.02,
             f'{value:.4f}', ha='center', va='top', fontweight='bold', fontsize=9, color='white')

for bar, value in zip(bars2_sd, sd_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 添加网格线
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)

# -------------------------- 整体布局调整与保存 --------------------------
plt.tight_layout(pad=3.0)  # 调整子图间距
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight', facecolor='white')  # 高分辨率保存
plt.show()