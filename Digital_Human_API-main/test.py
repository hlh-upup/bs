import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(24, 14))
ax.set_xlim(0, 24)
ax.set_ylim(0, 14)
ax.axis('off')

# 颜色配置（匹配原图）
colors = {
    'green': '#58D68D',   # 绿色管线背景
    'green_dark': '#27AE60',
    'orange': '#F8C471',  # 橙色管线背景
    'orange_dark': '#E67E22',
    'purple': '#BB8FCE',  # 紫色管线背景
    'purple_dark': '#8E44AD',
    'red': '#F1948A',     # 红色管线背景
    'red_dark': '#E74C3C',
    'gray': '#BFC9CA',    # 灰色管线背景
    'gray_dark': '#7F8C8D',
    'header': '#D7DBDD',  # 顶部和底部灰色
    'middle': '#D5DBDB'   # 中间层灰色
}

# 管线配置数据
pipelines = [
    {'name': '绿色管线', 'x': 4, 'color_bg': colors['green'], 'color_dark': colors['green_dark'],
     'steps': ['音频提取', '声学特征建模', '时序分帧\n(128维)', '音频嵌入']},
    {'name': '橙色管线', 'x': 8, 'color_bg': colors['orange'], 'color_dark': colors['orange_dark'],
     'steps': ['脸部裁剪', '3D人脸重建', '几何系数推导\n(64维)', '几何嵌入']},
    {'name': '紫色管线', 'x': 12, 'color_bg': colors['purple'], 'color_dark': colors['purple_dark'],
     'steps': ['关键点检测', '运动轨迹分析', '运动幅度计算\n(256维)', '运动嵌入']},
    {'name': '红色管线', 'x': 16, 'color_bg': colors['red'], 'color_dark': colors['red_dark'],
     'steps': ['动作单元识别', '表情强度编码', '表情对齐\n(128维)', '表情嵌入']},
    {'name': '灰色管线', 'x': 20, 'color_bg': colors['gray'], 'color_dark': colors['gray_dark'],
     'steps': ['全帧提取', '身份特征编码', '特征对齐\n(512维)', '身份嵌入']}
]

# 1. 绘制顶部输入框
input_box = FancyBboxPatch((1, 12), 22, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['header'], edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(12, 12.5, '输入视频流', ha='center', va='center', fontsize=16, fontweight='bold')

# 2. 绘制五个管线
y_positions = [10.2, 8.4, 6.6, 4.8]  # 四个步骤的Y坐标

for pipe in pipelines:
    x = pipe['x']
    bg_color = mcolors.to_rgba(pipe['color_bg'], alpha=0.6)
    
    # 绘制大背景框（带圆角）
    bg_box = FancyBboxPatch((x-1.8, 3.8), 3.6, 8.0, boxstyle="round,pad=0.15",
                           facecolor=bg_color, edgecolor=pipe['color_dark'], linewidth=3)
    ax.add_patch(bg_box)
    
    # 绘制左上角标签
    label = FancyBboxPatch((x-1.6, 11.2), 1.6, 0.5, boxstyle="round,pad=0.05",
                          facecolor=pipe['color_dark'], edgecolor='none')
    ax.add_patch(label)
    ax.text(x-0.8, 11.45, pipe['name'], ha='center', va='center', 
            fontsize=11, color='white', fontweight='bold')
    
    # 绘制四个步骤框
    for i, step in enumerate(pipe['steps']):
        y = y_positions[i]
        # 步骤框
        step_box = FancyBboxPatch((x-1.4, y-0.35), 2.8, 0.7, boxstyle="round,pad=0.08",
                                 facecolor='#F8F9F9', edgecolor='black', linewidth=1.2)
        ax.add_patch(step_box)
        ax.text(x, y, step, ha='center', va='center', fontsize=10.5)
        
        # 垂直箭头连接步骤
        if i < len(pipe['steps']) - 1:
            ax.annotate('', xy=(x, y_positions[i+1]+0.35), xytext=(x, y-0.35),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 从输入框到第一步骤的箭头
    ax.annotate('', xy=(x, y_positions[0]+0.35), xytext=(x, 12),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 从最后步骤到下方中间层的箭头
    ax.annotate('', xy=(x, 3.8), xytext=(x, y_positions[-1]-0.35),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 3. 绘制中间处理层（已去掉PCA表格）
# 左侧：模态嵌入连接
merge_box = FancyBboxPatch((2, 2.0), 4, 0.9, boxstyle="round,pad=0.1",
                          facecolor=colors['middle'], edgecolor='black', linewidth=1.5)
ax.add_patch(merge_box)
ax.text(4, 2.45, '模态嵌入连接', ha='center', va='center', fontsize=12, fontweight='bold')

# 右侧：PCA全局变换
pca_box = FancyBboxPatch((17, 2.0), 4, 0.9, boxstyle="round,pad=0.1",
                        facecolor=colors['middle'], edgecolor='black', linewidth=1.5)
ax.add_patch(pca_box)
ax.text(19, 2.45, 'PCA全局变换', ha='center', va='center', fontsize=12, fontweight='bold')

# 从各管线汇聚到"模态嵌入连接"的箭头
for pipe in pipelines:
    x = pipe['x']
    # 使用弧线连接
    rad = 0.15 if x > 6 else -0.15
    ax.annotate('', xy=(6, 2.45), xytext=(x, 3.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                             connectionstyle=f"arc3,rad={rad}"))

# 从"模态嵌入连接"到"PCA全局变换"的箭头（替代原来的表格）
ax.annotate('', xy=(17, 2.45), xytext=(6, 2.45),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 从"PCA全局变换"到输出框的箭头
ax.annotate('', xy=(19, 2.0), xytext=(19, 1.5),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 4. 绘制底部输出框
output_box = FancyBboxPatch((1, 0.5), 22, 1, boxstyle="round,pad=0.1",
                           facecolor=colors['header'], edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(12, 1.0, '367维 最终融合特征向量', ha='center', va='center', 
        fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_fusion_pipeline.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()