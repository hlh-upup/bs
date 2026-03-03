import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')

# 定义颜色
colors = {
    'audio': '#81C784',      # 浅绿色 - 语音
    'geo': '#FFB74D',        # 浅橙色 - 几何
    'motion': '#BA68C8',     # 浅紫色 - 运动
    'expr': '#F06292',       # 浅粉色 - 表情
    'fusion': '#BDBDBD',     # 灰色 - 融合
    'transformer': '#E0E0E0', # 浅灰 - Transformer
    'head_audio': '#81C784',
    'head_expr': '#FFB74D',
    'head_quality': '#BA68C8',
    'head_consistency': '#F06292',
    'head_overall': '#9E9E9E'
}

def draw_box(ax, x, y, width, height, color, text, text_color='black', fontsize=9, bold=True):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    
    lines = text.split('\n')
    line_height = 0.13
    start_y = y + (len(lines) - 1) * line_height / 2
    
    for i, line in enumerate(lines):
        weight = 'bold' if bold else 'normal'
        ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                fontsize=fontsize, color=text_color, weight=weight)

# ==================== 左侧：语音嵌层 ====================
ax.text(2.5, 10.3, '', fontsize=13, weight='bold', ha='center')

# 四个嵌入层
embed_y_positions = [8.8, 7.0, 5.2, 3.4]
embed_texts = [
    '语音嵌入\n(Audio Embedding,\n64维)',
    '几何嵌入\n(Geometric\nEmbedding,\n32维)',
    '运动嵌入\n(Motion Embedding\n128维)',
    '表情嵌入\n(Expression Embedding\n64维)'
]
embed_colors = [colors['audio'], colors['geo'], colors['motion'], colors['expr']]

for y, text, color in zip(embed_y_positions, embed_texts, embed_colors):
    draw_box(ax, 2.5, y, 2.4, 1.0, color, text, fontsize=9)

# ==================== 中间：融合+特征编码 ====================
ax.text(6.5, 10.3, '', fontsize=13, weight='bold', ha='center')

# 跨模态特征拼接
draw_box(ax, 6.5, 8.8, 3.2, 0.9, colors['fusion'], 
         '跨模态特征拼接与逐元素平均\n(Concatenation & Element-wise\nAverage, 367维)', fontsize=9)

# 可学习位置编码
draw_box(ax, 6.5, 7.0, 3.0, 0.9, colors['fusion'], 
         '可学习位置编码\n(Learnable Positional Encoding\n367维)', fontsize=9)

# 融合特征 + 位置编码
draw_box(ax, 6.5, 5.2, 3.0, 0.9, colors['fusion'], 
         '融合特征 + 位置编码\n(Fused Features + PE, 367维)', fontsize=9)

# ==================== 右侧：Transformer ====================
ax.text(11.5, 10.3, 'Transformer编码器', 
        fontsize=12, weight='bold', ha='center')

# Transformer外框
transformer_x = 11.5
transformer_y = 6.5
t_width = 3.8
t_height = 6.2

outer_box = FancyBboxPatch((transformer_x - t_width/2, transformer_y - t_height/2), 
                           t_width, t_height,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor='#F5F5F5', edgecolor='gray', linewidth=2)
ax.add_patch(outer_box)

# Layer n 标签
ax.text(10, 9.3, 'Layer n', fontsize=10, ha='center', style='italic')
ax.text(13.2, 9.3, 'n=1', fontsize=10, ha='center')
ax.text(13.2, 3.5, 'n=6', fontsize=10, ha='center')

# Transformer内部组件
components = [
    (8.8, '层归一化\n(LayerNorm)'),
    (7.8, '逐位置前馈网络\nNetwork'),
    (6.8, '层归一化'),
    (5.8, '层归一化'),
    (4.8, '多头自注意力子块\nMulti-Head'),
    (3.8, '层归一化')
]

for y, text in components:
    draw_box(ax, 11.5, y, 2.6, 0.7, colors['fusion'], text, fontsize=9)

# 垂直连接箭头
for i in range(len(components)-1):
    y1 = components[i][0] - 0.35
    y2 = components[i+1][0] + 0.35
    ax.annotate('', xy=(11.5, y2), xytext=(11.5, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


# 最下面一条残差连接（到多头自注意力子块前）
ax.plot([12.8, 13.1, 13.1], [3.8, 3.8, 4.8], 'k-', lw=1.5)
ax.annotate('', xy=(12.8, 4.8), xytext=(13.1, 4.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(13.2, 4.3, '残差连接', fontsize=8, rotation=90, va='center')

# 从上往下第三个层归一化到第一个层归一化的残差连接
ax.plot([12.8, 13.1, 13.1], [6.8, 6.8, 8.8], 'k-', lw=1.5)
ax.annotate('', xy=(12.8, 8.8), xytext=(13.1, 8.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(13.2, 7.8, '残差连接', fontsize=8, rotation=90, va='center')

# 多头自注意力左右箭头


# ==================== 最右侧：预测头 ====================
ax.text(15.8, 10.3, '', fontsize=13, weight='bold', ha='center')

# 五个预测头
head_data = [
    (8.8, colors['head_audio'], '唇形同步预测头\n[1.0, 5.0] 评分'),
    (7.6, colors['head_expr'], '表情自然度预测头\n[1.0, 5.0] 评分'),
    (6.4, colors['head_quality'], '音频质量预测头\n[1.0, 5.0] 评分'),
    (5.2, colors['head_consistency'], '跨模态一致性预测头\n[1.0, 5.0] 评分'),
    (4.0, colors['head_overall'], '整体感知质量预测头\n[1.0, 5.0] 评分')
]

for y, color, text in head_data:
    draw_box(ax, 15.8, y, 2.6, 0.9, color, text, fontsize=9)

# ==================== 连接箭头 ====================
# 左侧四个嵌入层到中间拼接框
for y in embed_y_positions:
    ax.annotate('', xy=(4.9, 8.8), xytext=(3.7, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 融合框之间的垂直流向
ax.annotate('', xy=(6.5, 7.45), xytext=(6.5, 8.35),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.annotate('', xy=(6.5, 5.65), xytext=(6.5, 6.55),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 跨模态平均到Transformer（并行直连）
ax.annotate('', xy=(9.6, 8.8), xytext=(8.1, 8.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8))

# 融合特征到Transformer
ax.annotate('', xy=(9.6, 5.2), xytext=(8.0, 5.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Transformer到各预测头
for y, _, _ in head_data:
    ax.annotate('', xy=(14.5, y), xytext=(13.4, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()