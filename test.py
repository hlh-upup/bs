import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# 更现代的高级字体配置 (优先使用微软雅黑)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 进一步加宽画布，增加模块间呼吸感
fig, ax = plt.subplots(figsize=(30, 15))
ax.set_xlim(0, 31)
ax.set_ylim(-1, 14.5)
ax.axis('off')

# 设计师级配色调色板 (低饱和度、高质感莫兰迪色系)
c = {
    'input': '#E3F2FD',        # 冰冷蓝
    'enc_out': '#F8F9FA',      # 极浅科技灰
    'enc_in': '#FFFFFF',       # 纯白
    'attention': '#E8EAF6',    # 浅群青紫
    'pooling': '#FFF3E0',      # 暖橙白
    'p1_stack': '#E8F5E9',     # 薄荷绿
    'p1_pos': '#FFF9C4',       # 暖鹅黄
    'p1_trans': '#FFF8E1',     # 更浅的暖黄
    'p2': '#E8F5E9',           # 薄荷绿
    'head': '#F3E5F5',         # 淡雅紫
    'edge': '#34495E',         # 深黛蓝 (比纯黑更优雅)
    'edge_dash': '#95A5A6',    # 银灰色 (用于虚线)
    'arrow': '#546E7A',        # 蓝灰色 (用于箭头)
    'text': '#2C3E50',         # 夜空文字黑
    'cite': '#7F8C8D',         # 注浆辅助灰
    'highlight': '#E74C3C'     # 强调红 (用于底部的突出提醒)
}

# 全局阴影特效，让元素有悬浮立体感
shadow = [pe.SimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace='black', alpha=0.08), pe.Normal()]

def draw_box(ax, x, y, w, h, text, facecolor, fontsize=10, bold=False, edgecolor=c['edge'], lw=1.5, zorder=20, text_ha='center', use_shadow=True):
    # 使用略大的 rounding_size 让边角更平滑现代
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder)
    if use_shadow:
        box.set_path_effects(shadow)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    if text:
        ax.text(x, y, text, ha=text_ha, va='center', fontsize=fontsize,
                color=c['text'], weight=weight, zorder=zorder+1, linespacing=1.4)

def draw_arrow(ax, start, end, style='-|>', color=c['arrow'], lw=1.5, zorder=10, conn="arc3,rad=0"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                           color=color, linewidth=lw,
                           connectionstyle=conn,
                           mutation_scale=14, zorder=zorder) # mutation_scale 适中，让箭头秀气一点
    ax.add_patch(arrow)

# ========== 核心坐标锚点 ==========
row_y = [11.5, 8.5, 5.5, 2.5]
col_input = 2.0
col_enc = 7.0
col_att = 12.0
col_pool = 15.5
col_fus = 20.0
col_avg = 23.0
col_gfv = 24.5
col_head = 28.5

# 1. 降维特征输入
input_titles = [
    '视觉特征\n(100维)',
    '音频特征\n(200维)',
    '面部关键点\n特征(50维)',
    '面部动作单元\n特征(17维)'
]
ax.text(col_input, 13.8, "降维特征输入", ha='center', va='bottom', fontsize=12, weight='bold', color=c['text'])

for i, y in enumerate(row_y):
    draw_box(ax, col_input, y, 2.2, 1.6, input_titles[i], c['input'], fontsize=10, bold=True)
    if i == 3:
        ax.text(col_input, y - 1.3, "", ha='center', va='center', fontsize=9, color=c['cite'])

# 2. 模态专属残差编码器
ax.text(col_enc, 13.8, "模态专属残差编码器 (3层)", ha='center', va='bottom', fontsize=12, weight='bold', color=c['text'])

enc_w = 4.6
for i, y in enumerate(row_y):
    # 外框圆角稍微放软
    draw_box(ax, col_enc, y, enc_w, 2.3, "", c['enc_out'], lw=1.2, edgecolor=c['edge_dash'])
    ax.text(col_enc, y - 1.45, "", ha='center', va='center', fontsize=9, color=c['cite'])
    
    # 内部组件
    draw_box(ax, col_enc - 1.2, y, 1.4, 1.3, "残差特征\n编码器", c['enc_in'], fontsize=9.5)
    draw_box(ax, col_enc + 0.6, y + 0.7, 1.4, 0.45, "线性层", c['enc_in'], fontsize=8.5)
    draw_box(ax, col_enc + 0.6, y, 1.4, 0.55, "层归一化\n-> ReLU", c['enc_in'], fontsize=8.5)
    draw_box(ax, col_enc + 0.6, y - 0.7, 1.4, 0.45, "Dropout", c['enc_in'], fontsize=8.5)
    
    # 编码器内部连线
    draw_arrow(ax, (col_enc - 0.5, y), (col_enc - 0.1, y), lw=1.2)
    
    # 捷径连接 (更平滑的弧线)
    draw_arrow(ax, (col_enc - 0.3, y), (col_enc - 0.3, y + 1.25), lw=1.2, style='-')
    draw_arrow(ax, (col_enc - 0.3, y + 1.25), (col_enc + 1.7, y + 1.25), lw=1.2, style='-')
    draw_arrow(ax, (col_enc + 1.7, y + 1.25), (col_enc + 1.7, y + 0.3), lw=1.2, style='-|>')
    ax.text(col_enc + 1.8, y, "残差捷径连接", ha='left', va='center', fontsize=8.5, color=c['arrow'])

# 输入到编码器的箭头
for y in row_y:
    draw_arrow(ax, (col_input + 1.1, y), (col_enc - enc_w/2 - 0.1, y))

# 3. 自注意力模块
ax.text(col_att, 13.8, "共享权重时间自注意力模块", ha='center', va='bottom', fontsize=12, weight='bold', color=c['text'])
draw_box(ax, col_att, 7.0, 2.2, 11.4, "", c['attention'])
att_text = "8头\n多头\n自注意力\n+ FFN\n+\n残差\n+ 层归一化\n"
ax.text(col_att, 7.0, att_text, ha='center', va='center', fontsize=12, weight='bold', zorder=25, linespacing=1.5, color=c['text'])

# 编码器到注意力的连接
for y in row_y:
    draw_arrow(ax, (col_enc + enc_w/2 + 0.1, y), (col_att - 1.1, y))
    ax.text(col_att - 1.8, y + 0.2, "512", ha='center', va='bottom', fontsize=10, color=c['text'])

# 4. 池化层
for y in row_y:
    draw_box(ax, col_pool, y, 2.8, 1.5, "均值-最大\n双池化\n与线性投影", c['pooling'], fontsize=9.5)
    ax.text(col_pool, y - 1.0, "", ha='center', va='center', fontsize=8, zorder=30, color=c['cite'])

# 注意力到池化的连接
for y in row_y:
    draw_arrow(ax, (col_att + 1.1, y), (col_pool - 1.4, y), lw=1.5, zorder=30)
    ax.text(col_att + 1.8, y + 0.2, "512", ha='center', va='bottom', fontsize=10, color=c['text'])

# 5. 双路径融合网络
ax.text(col_fus, 13.8, "双路径融合", ha='center', va='bottom', fontsize=13, weight='bold', color=c['text'])

p1_y_center = 9.2
p1_w = 4.0
draw_box(ax, col_fus, p1_y_center, p1_w, 6.2, "", 'none', edgecolor=c['edge_dash'], lw=1.8, use_shadow=False)
patch_p1 = [p for p in ax.patches][-1]
patch_p1.set_linestyle('--')
# 为 Path1 标签加上高雅的设计
ax.text(col_fus - p1_w/2 + 0.7, p1_y_center + 2.8, " 路径 1 ", ha='center', va='center', fontsize=11, weight='bold', zorder=25, 
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=c['edge_dash'], alpha=0.9))

p2_y_center = 3.5
draw_box(ax, col_fus, p2_y_center, p1_w, 3.2, "", 'none', edgecolor=c['edge_dash'], lw=1.8, use_shadow=False)
patch_p2 = [p for p in ax.patches][-1]
patch_p2.set_linestyle('--')
ax.text(col_fus - p1_w/2 + 0.7, p2_y_center + 1.3, " 路径 2 ", ha='center', va='center', fontsize=11, weight='bold', zorder=25,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=c['edge_dash'], alpha=0.9))

# Path 1 内部组件
y_p1_1 = 11.5
y_p1_2 = 10.2
y_p1_3 = 8.5
y_p1_4 = 6.8

draw_box(ax, col_fus, y_p1_1, 3.4, 1.1, "模态维度堆叠\n(4x512矩阵)", c['p1_stack'], fontsize=9.5)
draw_box(ax, col_fus, y_p1_2, 2.4, 0.9, "可学习\n位置编码", c['p1_pos'], fontsize=9.5)
draw_box(ax, col_fus, y_p1_3, 3.4, 1.9, "多层\nTransformer\n编码器\n(全局自注意力)", c['p1_trans'], fontsize=9.5)
ax.text(col_fus + 1.7, y_p1_3 - 0.75, "", ha='left', va='top', fontsize=8, color=c['cite'])
draw_box(ax, col_fus, y_p1_4, 2.4, 0.9, "均值池化", c['p1_stack'], fontsize=9.5)

draw_arrow(ax, (col_fus, y_p1_1 - 0.55), (col_fus, y_p1_2 + 0.45))
draw_arrow(ax, (col_fus, y_p1_2 - 0.45), (col_fus, y_p1_3 + 0.95))
draw_arrow(ax, (col_fus, y_p1_3 - 0.95), (col_fus, y_p1_4 + 0.45))

# Path 2 内部组件
draw_box(ax, col_fus, p2_y_center, 3.4, 2.2, "多尺度融合网络\n(投影 -> 拼接 ->\n 2层全连接)", c['p2'], fontsize=9.5)
ax.text(col_fus, p2_y_center - 1.3, "", ha='center', va='top', fontsize=8, color=c['cite'])


# 汇流排 (Bus-line) 连接逻辑优化
bus_x = 17.6
# 虚线线干
ax.plot([bus_x, bus_x], [2.5, 11.5], linestyle='--', color=c['arrow'], zorder=5, lw=1.8)
for i, y in enumerate(row_y):
    draw_arrow(ax, (col_pool + 1.4, y), (bus_x, y), style='-', lw=1.8)
    ax.text(col_pool + 1.7, y + 0.2, "512", ha='center', va='bottom', fontsize=9.5, color=c['text'])

draw_arrow(ax, (bus_x, y_p1_1), (col_fus - 1.7, y_p1_1), lw=1.8)
draw_arrow(ax, (bus_x, p2_y_center), (col_fus - 1.7, p2_y_center), lw=1.8)

# 6. 后处理与输出模块
draw_box(ax, col_avg, 7.0, 1.4, 1.0, "均值融合", c['enc_in'], lw=1.5, fontsize=10.5, bold=True)

# 动态计算 Path 1 & Path 2 右边缘至 Averge 
path1_right = col_fus + 1.2  # 均值池化的一半宽
path2_right = col_fus + 1.7  # 多尺度融合的一半宽

# 用优美的贝塞尔曲线连接
draw_arrow(ax, (path1_right, y_p1_4), (col_avg - 0.7, 7.3), conn="arc3,rad=-0.15", lw=1.5)
draw_arrow(ax, (path2_right, p2_y_center), (col_avg - 0.7, 6.7), conn="arc3,rad=0.15", lw=1.5)

ax.text(col_fus + 2.1, 7.5, "512维", ha='center', va='bottom', fontsize=9.5, color=c['cite'])
ax.text(col_fus + 2.1, 5.5, "512维", ha='center', va='top', fontsize=9.5, color=c['cite'])

draw_box(ax, col_gfv, 7.0, 1.0, 3.2, "全局\n融合特征\n向量", c['attention'], fontsize=10, bold=True)
draw_arrow(ax, (col_avg + 0.7, 7.0), (col_gfv - 0.5, 7.0), lw=1.8)
ax.text(col_avg + 1.0, 7.2, "512维", ha='center', va='bottom', fontsize=10, weight='bold', color=c['text'])

# 7. 预测头
ax.text(col_head, 13.8, "改进的任务预测头", ha='center', va='bottom', fontsize=12, weight='bold', color=c['text'])

heads_y = [11.5, 9.25, 7.0, 4.75, 2.5]
head_w = 4.8
head_names = ['唇形同步', '面部表情', '音频质量', '跨模态匹配', '综合表现']
for i, y in enumerate(heads_y):
    txt = "可学习特征缩放 ->\n线性层+层归一化+GELU+Dropout\n线性层+层归一化+GELU\n线性层 -> Sigmoid\n-> 仿射变换 -> [1.0, 5.0]"
    draw_box(ax, col_head, y, head_w, 2.0, txt, c['head'], fontsize=9.5)
    
    # 从全局向量发散到 Heads
    draw_arrow(ax, (col_gfv + 0.5, 7.0), (col_head - head_w/2 - 0.1, y), conn=f"arc3,rad={0.06 * (7.0 - y)}")
    
    # 结果标签
    label = head_names[i]
    draw_arrow(ax, (col_head + head_w/2 + 0.1, y), (col_head + head_w/2 + 0.8, y))
    ax.text(col_head + head_w/2 + 0.9, y, f"{label}", ha='left', va='center', fontsize=11, weight='bold', color=c['edge'])

# 8. 底部损失函数区
b_y = -0.4
b_line_y = 0.4

# 横向基础轴线
ax.plot([col_enc, col_fus], [b_line_y, b_line_y], '-', color=c['edge'], lw=1.8, zorder=5)

b_loss1 = "★ MSE + L1 混合损失及可学习Sigmoid权重"
ax.text((col_enc + col_att)/2, b_y, b_loss1, ha='center', va='center', fontsize=12, weight='bold', color=c['highlight'])

b_loss2 = "★ 跨任务一致性正则化损失"
ax.text((col_pool + col_fus)/2 + 0.5, b_y, b_loss2, ha='center', va='center', fontsize=12, weight='bold', color=c['highlight'])

b_loss3 = "*. 针对缺失标签的标签掩码机制"
ax.text(30.6, b_y, b_loss3, ha='right', va='center', fontsize=10.5, color=c['cite'], style='italic')

# 指向各模块的支撑箭头
draw_arrow(ax, (col_enc, b_line_y), (col_enc, 1.45), lw=1.8)
draw_arrow(ax, (col_att, b_line_y), (col_att, 1.45), lw=1.8)
draw_arrow(ax, (col_fus, b_line_y), (col_fus, 1.95), lw=1.8)

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
