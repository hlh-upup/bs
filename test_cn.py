import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(28, 14))
ax.set_xlim(0, 31)
ax.set_ylim(-1, 14.5)
ax.axis('off')

# Colors matching the original image as closely as possible
c_bg = '#F2F2F2'       # Input grey
c_enc_out = '#E8ECE8'  # Encoder outer grey
c_att = '#CEDFF1'      # Attention light blue
c_pool = '#F1DBD0'     # Pooling light pink/orange
c_p1_stack = '#E1EED3' # Path1 light green
c_p1_pos = '#FBE7D2'   # Path1 light orange
c_p1_trans = '#FDF0CD' # Path1 light yellow
c_p2 = '#E1EED3'       # Path2 light green
c_head = '#E1EED3'     # Head light green
text_c = 'black'

def draw_box(ax, x, y, w, h, text, facecolor, fontsize=10, bold=False, edgecolor='black', lw=1.2, zorder=20, text_ha='center'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    if text:
        ax.text(x, y, text, ha=text_ha, va='center', fontsize=fontsize,
                color=text_c, weight=weight, zorder=zorder+1)

def draw_arrow(ax, start, end, style='-|>', color='black', lw=1.5, zorder=10, conn="arc3,rad=0"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                           color=color, linewidth=lw,
                           connectionstyle=conn,
                           mutation_scale=15, zorder=zorder)
    ax.add_patch(arrow)

# Centers
row_y = [11.5, 8.5, 5.5, 2.5]
col_input = 2.0
col_enc = 7.0
col_att = 12.0
col_pool = 15.5
col_fus = 20.0
col_avg = 23.0
col_gfv = 24.5
col_head = 28.5

# 1. Inputs
input_titles = [
    '视觉特征\n(100维)',
    '音频特征\n(200维)',
    '面部关键点\n特征(50维)',
    '面部动作单元\n特征(17维)'
]
ax.text(col_input, 13.7, "降维特征输入\n[cite: 3.2.1, 3.2.2]", ha='center', va='bottom', fontsize=11, weight='bold')

for i, y in enumerate(row_y):
    draw_box(ax, col_input, y, 2.2, 1.6, input_titles[i], c_bg, fontsize=10, bold=True)
    if i == 3:
        ax.text(col_input, y - 1.2, "[cite: 3.2.2]", ha='center', va='center', fontsize=9)

# 2. Encoders
ax.text(col_enc, 13.7, "模态专属残差编码器 (3层)\n[cite: 3.2.3]", ha='center', va='bottom', fontsize=11, weight='bold')

enc_w = 4.6
for i, y in enumerate(row_y):
    # Outer box
    draw_box(ax, col_enc, y, enc_w, 2.2, "", c_enc_out, lw=1.5)
    ax.text(col_enc, y - 1.4, "[cite: 3.2.3]", ha='center', va='center', fontsize=9)
    
    # Inner boxes
    c_in = 'white'
    draw_box(ax, col_enc - 1.2, y, 1.4, 1.2, "残差特征\n编码器", c_in, fontsize=9)
    draw_box(ax, col_enc + 0.5, y + 0.6, 1.3, 0.4, "线性层", c_in, fontsize=8)
    draw_box(ax, col_enc + 0.5, y, 1.3, 0.5, "层归一化\n-> ReLU", c_in, fontsize=8)
    draw_box(ax, col_enc + 0.5, y - 0.6, 1.3, 0.4, "Dropout", c_in, fontsize=8)
    
    # Arrows inside encoder
    draw_arrow(ax, (col_enc - 0.5, y), (col_enc - 0.1, y), lw=1.0)
    
    # Shortcut arrow
    draw_arrow(ax, (col_enc - 0.3, y), (col_enc - 0.3, y + 1.2), lw=1.0, style='-')
    draw_arrow(ax, (col_enc - 0.3, y + 1.2), (col_enc + 1.6, y + 1.2), lw=1.0, style='-')
    draw_arrow(ax, (col_enc + 1.6, y + 1.2), (col_enc + 1.6, y + 0.2), lw=1.0, style='-|>')
    ax.text(col_enc + 1.65, y, "残差捷径连接", ha='left', va='center', fontsize=8)

# Input to Encoder Arrows
for y in row_y:
    draw_arrow(ax, (col_input + 1.1, y), (col_enc - enc_w/2 - 0.1, y))

# 3. Attention
ax.text(col_att, 13.7, "共享权重时间自注意力模块\n[cite: 3.2.3]", ha='center', va='bottom', fontsize=11, weight='bold')
draw_box(ax, col_att, 7.0, 2.0, 11.2, "", c_att)
att_text = "8头\n多头\n自注意力\n+ FFN\n+\n残差\n+ 层归一化\n\n[cite: 3.2.3]"
ax.text(col_att, 7.0, att_text, ha='center', va='center', fontsize=11, weight='bold', zorder=25)

# Encoder to Attention Arrows
for y in row_y:
    draw_arrow(ax, (col_enc + enc_w/2 + 0.1, y), (col_att - 1.0, y))
    ax.text(col_att - 1.8, y + 0.2, "512", ha='center', va='bottom', fontsize=10)

# 4. Pooling
for y in row_y:
    draw_box(ax, col_pool, y, 2.6, 1.4, "Mean-Max\nDouble Pooling\n& 线性层\nProjection", c_pool, fontsize=9)
    ax.text(col_pool, y - 0.9, "[cite: 3.2.3]", ha='center', va='center', fontsize=8, zorder=30)

# Attention to Pooling Arrows
for y in row_y:
    # VERY EXPLICIT horizontal multi-head attention to pooling arrows!
    draw_arrow(ax, (col_att + 1.0, y), (col_pool - 1.3, y), lw=1.5, zorder=30)
    ax.text(col_att + 1.8, y + 0.2, "512", ha='center', va='bottom', fontsize=10)

# 5. 双路径融合
ax.text(col_fus, 13.7, "双路径融合", ha='center', va='bottom', fontsize=12, weight='bold')

# Dashed boxes for 路径 1 and 2
p1_y_center = 9.2
p1_w = 3.8
draw_box(ax, col_fus, p1_y_center, p1_w, 6.0, "", 'none', edgecolor='black', lw=1.5)
patch_p1 = [p for p in ax.patches][-1]
patch_p1.set_linestyle('--')
ax.text(col_fus - p1_w/2 + 0.5, p1_y_center + 2.7, "路径 1", ha='center', va='center', fontsize=10, weight='bold', zorder=25, bbox=dict(facecolor='white', edgecolor='none', pad=1))

p2_y_center = 3.5
draw_box(ax, col_fus, p2_y_center, p1_w, 3.0, "", 'none', edgecolor='black', lw=1.5)
patch_p2 = [p for p in ax.patches][-1]
patch_p2.set_linestyle('--')
ax.text(col_fus - p1_w/2 + 0.5, p2_y_center + 1.2, "路径 2", ha='center', va='center', fontsize=10, weight='bold', zorder=25, bbox=dict(facecolor='white', edgecolor='none', pad=1))

# 路径 1 Components
y_p1_1 = 11.5
y_p1_2 = 10.3
y_p1_3 = 8.6
y_p1_4 = 6.9

draw_box(ax, col_fus, y_p1_1, 3.2, 1.0, "模态维度堆叠\n(4x512矩阵)", c_p1_stack, fontsize=9)
draw_box(ax, col_fus, y_p1_2, 2.2, 0.8, "可学习\n位置编码", c_p1_pos, fontsize=9)
draw_box(ax, col_fus, y_p1_3, 3.2, 1.8, "多层\nTransformer\n编码器\n(全局自注意力)", c_p1_trans, fontsize=9)
ax.text(col_fus + 1.6, y_p1_3 - 0.7, "[cite: 3.2.3]", ha='left', va='top', fontsize=8)
draw_box(ax, col_fus, y_p1_4, 2.2, 0.8, "均值池化", c_p1_stack, fontsize=9)

draw_arrow(ax, (col_fus, y_p1_1 - 0.5), (col_fus, y_p1_2 + 0.4))
draw_arrow(ax, (col_fus, y_p1_2 - 0.4), (col_fus, y_p1_3 + 0.9))
draw_arrow(ax, (col_fus, y_p1_3 - 0.9), (col_fus, y_p1_4 + 0.4))

# 路径 2 Components
draw_box(ax, col_fus, p2_y_center, 3.2, 2.0, "多尺度融合网络\n(投影 -> 拼接 ->\n 2层全连接)", c_p2, fontsize=9)
ax.text(col_fus, p2_y_center - 1.2, "[cite: 3.2.3]", ha='center', va='top', fontsize=8)

# Pooling to Fusion Arrows
bus_x = 17.6
# Dashed bus line
ax.plot([bus_x, bus_x], [2.5, 11.5], linestyle='--', color='black', zorder=5, lw=1.5)
for i, y in enumerate(row_y):
    draw_arrow(ax, (col_pool + 1.3, y), (bus_x, y), style='-')
    ax.text(col_pool + 1.7, y + 0.2, "512", ha='center', va='bottom', fontsize=9)

# Arrow from bus to Stack (路径 1)
draw_arrow(ax, (bus_x, 11.5), (col_fus - 1.6, 11.5))
# Arrow from bus to 路径 2
draw_arrow(ax, (bus_x, p2_y_center), (col_fus - 1.6, p2_y_center))


# 均值融合 and Global Vector
draw_box(ax, col_avg, 7.0, 1.2, 0.8, "均值融合", 'white', lw=1.5)

draw_arrow(ax, (col_fus + 1.9, y_p1_4), (col_avg - 0.6, 7.2), conn="arc3,rad=-0.1")
draw_arrow(ax, (col_fus + 1.9, p2_y_center), (col_avg - 0.6, 6.8), conn="arc3,rad=0.1")
ax.text(col_fus + 2.1, y_p1_4 + 0.2, "512", ha='left', va='bottom', fontsize=9)
ax.text(col_fus + 2.1, p2_y_center + 0.2, "512", ha='left', va='bottom', fontsize=9)

draw_box(ax, col_gfv, 7.0, 0.9, 3.0, "全局\n融合特征\n向量\n[cite: 3.2.3]", c_att, fontsize=9)
draw_arrow(ax, (col_avg + 0.6, 7.0), (col_gfv - 0.45, 7.0))
ax.text(col_avg + 0.9, 7.2, "512维", ha='center', va='bottom', fontsize=9, weight='bold')

# 6. Prediction Heads
ax.text(col_head, 13.7, "改进的任务预测头\n[cite: 3.2.1, 3.2.3]", ha='center', va='bottom', fontsize=11, weight='bold')

heads_y = [11.5, 9.25, 7.0, 4.75, 2.5]
head_w = 4.8
head_names = ['唇形同步', '面部表情', '音频质量', '跨模态匹配', '综合表现']
for i, y in enumerate(heads_y):
    txt = "Learnable Feature Scaling ->\n线性层+LayerNorm+GELU+Dropout\n线性层+LayerNorm+GELU\n线性层 -> Sigmoid\n-> Affine Transform -> [1.0, 5.0]"
    draw_box(ax, col_head, y, head_w, 2.0, txt, c_head, fontsize=9)
    
    # Arrow from global to head
    draw_arrow(ax, (col_gfv + 0.45, 7.0), (col_head - head_w/2 - 0.1, y), conn=f"arc3,rad={0.05 * (7.0 - y)}")
    
    # Output labels
    label = head_names[i]
    draw_arrow(ax, (col_head + head_w/2 + 0.1, y), (col_head + head_w/2 + 0.7, y))
    ax.text(col_head + head_w/2 + 0.8, y, f"{label}\n[cite: 3.2.1, 3.2.3]", ha='left', va='center', fontsize=10, weight='bold')

# Bottom Text
b_y = -0.5
b_loss1 = "MSE + L1 混合损失及可学习Sigmoid权重 [cite: 3.2.3]"
ax.text(col_enc + 0.5, b_y, b_loss1, ha='center', va='center', fontsize=11, weight='bold')

b_loss2 = "跨任务一致性正则化损失 [cite: 3.2.3]"
ax.text(col_pool + 1.0, b_y, b_loss2, ha='center', va='center', fontsize=11, weight='bold')

b_loss3 = "*. 针对缺失标签的标签掩码机制 [cite: 3.2.3]"
ax.text(30.5, b_y, b_loss3, ha='right', va='center', fontsize=10)

# Up arrows from bottom text
# Arrow to encoder
draw_arrow(ax, (col_enc, b_y + 0.3), (col_enc, 1.4), lw=1.2)
# Arrow to attention
draw_arrow(ax, (col_att, b_y + 0.3), (col_att, 1.4), lw=1.2)
# Arrow to fusion dashed box
draw_arrow(ax, (col_pool + 1.0, b_y + 0.3), (col_fus, 1.4), conn="arc3,rad=-0.1", lw=1.2)

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
