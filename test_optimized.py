import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 扩大画布以释放更多空间
fig, ax = plt.subplots(figsize=(26, 15))
ax.set_xlim(0, 26)
ax.set_ylim(-1, 14)
ax.axis('off')

# 更加现代化和平滑的颜色方案
colors = {
    'input': '#E3F2FD',      # Light blue
    'encoder': '#FFF3E0',    # Light orange
    'attention': '#EDE7F6',  # Light purple
    'pooling': '#E8F5E9',    # Light green
    'path1': '#FFF8E1',      # Light amber
    'path2': '#E0F7FA',      # Light cyan
    'fusion': '#FCE4EC',     # Light pink
    'head': '#F3E5F5',       # Very light purple
    'arrow': '#546E7A',
    'text': '#263238',
    'highlight': '#FF5722'
}

def draw_box(ax, x, y, width, height, text, color, fontsize=9, text_color='#263238', bold=False, alpha=1.0):
    """绘制带文本的矩形框,具有更现代的风格"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=color, edgecolor='#78909C', linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#546E7A', arrowstyle='-|>', linewidth=1.5, connectionstyle="arc3,rad=0"):
    """绘制更加美观的箭头"""
    arrow = FancyArrowPatch(start, end, arrowstyle=arrowstyle,
                           color=color, linewidth=linewidth,
                           connectionstyle=connectionstyle,
                           mutation_scale=15)
    ax.add_patch(arrow)

# ======== 布局参数 ========
col_x = [1.5, 5.0, 9.0, 12.5, 17.0, 23.5]  # X轴列位置
row_y = [11.5, 8.5, 5.5, 2.5]              # 输入模态Y轴位置

# 1. 输入模块 (Reduced-Dimension Feature Input)
input_labels = ['视觉特征\n(100维)', '音频特征\n(200维)', '面部关键点\n特征(50维)', '面部动作单元\n(AU)特征(17维)']
for i, y in enumerate(row_y):
    draw_box(ax, col_x[0], y, 2.2, 1.2, input_labels[i], colors['input'], fontsize=10, bold=True)
    ax.text(col_x[0], y - 0.8, '[cite: 3.2.2]', ha='center', va='top', fontsize=8, style='italic', color='#546E7A')

ax.text(col_x[0], 13.5, '降维特征输入', ha='center', va='bottom', fontsize=12, weight='bold', color=colors['text'])

# 2. 模态特定残差编码器
for i, y in enumerate(row_y):
    # 外框
    rect = mpatches.Rectangle((col_x[1] - 1.4, y - 1.4), 2.8, 2.8,
                              linewidth=1.2, edgecolor='#9E9E9E', facecolor='#FAFAFA',
                              linestyle='--', joinstyle='round', zorder=0)
    ax.add_patch(rect)
    
    # 内部层
    draw_box(ax, col_x[1], y + 0.8, 1.2, 0.4, '线性层', '#FFFFFF', fontsize=8)
    draw_box(ax, col_x[1], y + 0.2, 1.2, 0.5, '层归一化\n→ReLU', '#FFFFFF', fontsize=8)
    draw_box(ax, col_x[1], y - 0.4, 1.2, 0.4, 'Dropout层', '#FFFFFF', fontsize=8)
    
    # 残差捷径
    ax.plot([col_x[1] + 0.7, col_x[1] + 1.1, col_x[1] + 1.1, col_x[1] + 0.7], 
            [y + 0.8, y + 0.8, y - 0.4, y - 0.4], color='#78909C', linewidth=1.2)
    ax.text(col_x[1] + 1.15, y + 0.2, '残差\n连接', ha='left', va='center', fontsize=7, color='#546E7A')
    
    ax.text(col_x[1] - 1.6, y, '×3层', ha='right', va='center', fontsize=9, weight='bold', color=colors['highlight'])

ax.text(col_x[1], 13.5, '模态特定\n残差编码器', ha='center', va='bottom', fontsize=12, weight='bold', color=colors['text'])
ax.text(col_x[1], 12.8, '[cite: 3.2.3]', ha='center', va='bottom', fontsize=9, style='italic', color='#546E7A')

# 输入到编码器的箭头
for y in row_y:
    draw_arrow(ax, (col_x[0] + 1.1, y), (col_x[1] - 1.4, y))

# 3. 共享权重时间自注意力模块
att_height = 10.5
draw_box(ax, col_x[2], 7.0, 2.6, att_height, '', colors['attention'], alpha=0.9)
ax.text(col_x[2], 9.5, '8头多头\n时间自注意力', ha='center', va='center', fontsize=11, weight='bold')
ax.text(col_x[2], 8.0, '+ FFN', ha='center', va='center', fontsize=11, weight='bold')
ax.text(col_x[2], 6.5, '+ 残差', ha='center', va='center', fontsize=11, weight='bold')
ax.text(col_x[2], 5.0, '+ 层归一化', ha='center', va='center', fontsize=11, weight='bold')

ax.text(col_x[2], 13.5, '共享权重\n时间自注意力模块', ha='center', va='bottom', fontsize=12, weight='bold', color=colors['text'])

# 编码器到注意力的箭头
for y in row_y:
    draw_arrow(ax, (col_x[1] + 1.4, y), (col_x[2] - 1.3, y))
    ax.text(col_x[2] - 1.8, y + 0.2, '512维', ha='center', va='bottom', fontsize=8, color='#546E7A')

# 4. 池化投影层
for y in row_y:
    draw_box(ax, col_x[3], y, 2.4, 1.2, '均值-最大双池化\n与线性投影', colors['pooling'], fontsize=9)
    ax.text(col_x[3] + 1.5, y, '512维', ha='right', va='center', fontsize=8, weight='bold', color='#FF9800')

# 注意力到池化的箭头
for y in row_y:
    draw_arrow(ax, (col_x[2] + 1.3, y), (col_x[3] - 1.2, y))

ax.text(col_x[3], 13.5, '特征池化\n与维度统一', ha='center', va='bottom', fontsize=12, weight='bold')

# 5. 双路径融合网络
# Path 1
path1_y = 9.5
rect_p1 = mpatches.Rectangle((col_x[4] - 2.0, path1_y - 2.8), 4.0, 5.6,
                             linewidth=1.2, edgecolor='#9E9E9E', facecolor='#FAFAFA', linestyle='--')
ax.add_patch(rect_p1)
ax.text(col_x[4] - 2.2, path1_y, '路径1\n全局交互', ha='right', va='center', fontsize=10, weight='bold')

draw_box(ax, col_x[4], path1_y + 1.8, 3.2, 0.8, '模态维度堆叠 (4×512矩阵)', colors['path1'], fontsize=9)
draw_box(ax, col_x[4], path1_y + 0.6, 2.8, 0.6, '可学习位置编码', '#FFFFFF', fontsize=9)
draw_box(ax, col_x[4], path1_y - 0.6, 3.2, 1.2, '多层Transformer编码器\n(模态间全局自注意力)', colors['path1'], fontsize=9)
draw_box(ax, col_x[4], path1_y - 1.9, 2.0, 0.6, '均值池化', colors['path1'], fontsize=9)

# 箭头连接池化到 Path 1
for y in row_y:
    draw_arrow(ax, (col_x[3] + 1.2, y), (col_x[4]-2.0, path1_y + 1.8), connectionstyle=f"arc3,rad={-0.1 * (y - 7)}")

# Path 2
path2_y = 4.0
rect_p2 = mpatches.Rectangle((col_x[4] - 2.0, path2_y - 2.0), 4.0, 4.0,
                             linewidth=1.2, edgecolor='#9E9E9E', facecolor='#FAFAFA', linestyle='--')
ax.add_patch(rect_p2)
ax.text(col_x[4] - 2.2, path2_y, '路径2\n多尺度', ha='right', va='center', fontsize=10, weight='bold')

draw_box(ax, col_x[4], path2_y, 3.2, 2.2, '多尺度融合网络\n(特征投影 → 深度拼接\n→ 2层全连接网络)', colors['path2'], fontsize=9)

# 箭头连接池化到 Path 2 (聚合)
draw_arrow(ax, (col_x[3] + 1.2, 5.5), (col_x[4]-2.0, path2_y), connectionstyle="arc3,rad=0.1")
draw_arrow(ax, (col_x[3] + 1.2, 2.5), (col_x[4]-2.0, path2_y), connectionstyle="arc3,rad=-0.1")

# Average
avg_x = 19.5
avg_y = 7.0
draw_box(ax, avg_x, avg_y, 1.5, 1.0, '平均融合', colors['fusion'], fontsize=10, bold=True)
draw_arrow(ax, (col_x[4] + 2.0, path1_y - 1.9), (avg_x - 0.75, avg_y + 0.2))
draw_arrow(ax, (col_x[4] + 2.0, path2_y), (avg_x - 0.75, avg_y - 0.2))

draw_box(ax, avg_x + 2.0, avg_y, 1.4, 3.0, '全局融合\n特征向量\n(512维)', colors['fusion'], fontsize=9, bold=True)
draw_arrow(ax, (avg_x + 0.75, avg_y), (avg_x + 1.3, avg_y))

ax.text(col_x[4], 13.5, '双路径交互与融合', ha='center', va='bottom', fontsize=12, weight='bold')

# 6. 任务预测头
heads_y = [11.5, 9.25, 7.0, 4.75, 2.5]
head_labels = ['唇形同步', '面部表情', '音频质量预期', '跨模态匹配度', '整体拟真度']

for i, y in enumerate(heads_y):
    # 头框
    box_w = 4.2
    draw_box(ax, col_x[5], y, box_w, 1.6, '', colors['head'])
    
    # 内部
    ax.text(col_x[5], y + 0.3, '缩放→层归一化→GELU→Dropout\n线性层→Sigmoid→仿射缩放', ha='center', va='center', fontsize=7, color='#616161')
    
    # 标签
    ax.text(col_x[5] + box_w/2 + 0.2, y, head_labels[i], ha='left', va='center', fontsize=10, weight='bold')

    # 从融合特征连接到每个头
    draw_arrow(ax, (avg_x + 2.7, avg_y), (col_x[5] - box_w/2, y), connectionstyle=f"arc3,rad={0.05 * (avg_y - y)}")

ax.text(col_x[5], 13.5, '改进的任务预测头\n(含动态特征缩放)', ha='center', va='bottom', fontsize=12, weight='bold')

# 损失与注释
ax.text(4.0, 0.0, '★ MSE + L1 混合损失 (含可学习动态Sigmoid权重)', ha='center', va='center', fontsize=10, weight='bold', color='#D84315')
ax.text(12.0, 0.0, '★ 跨任务一致性正则化损失 (Cross-Task Consistency)', ha='center', va='center', fontsize=10, weight='bold', color='#D84315')
ax.text(21.0, 0.0, '* 动态标签掩码机制处理缺失数据 (Label Masking)', ha='left', va='center', fontsize=10, style='italic', color='#546E7A')

plt.tight_layout()
plt.savefig('model_architecture_optimized.png', dpi=300, bbox_inches='tight', facecolor='white')

# 如果你需要直接查看图片且环境支持弹窗可以取消注释下面的代码
# plt.show()
