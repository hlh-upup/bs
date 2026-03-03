import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os

# 高级中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(24, 15))
ax.set_xlim(0, 24)
ax.set_ylim(-3, 14.5)
ax.axis('off')

# 现代莫兰迪配色主题
colors = {
    'green': '#A8D8B9', 'green_dark': '#6EB98D',
    'orange': '#F3C5A5', 'orange_dark': '#D98A5B',
    'purple': '#C4B5E3', 'purple_dark': '#8D7BBE',
    'red': '#F3A6A6', 'red_dark': '#D26C6C',
    'blue': '#A6C8F3', 'blue_dark': '#6B9ED1',
    'header': '#EFEFEF', 'middle': '#DDE4EA', 'shadow': '#D0D0D0'
}

pipelines = [
    {'name': '音频特征', 'x': 4.8, 'bg': colors['green'], 'dark': colors['green_dark'],
     'steps': ['音频流提取', '声学特征建模', '时序分帧提取\n(128维)', '音频序列嵌入']},
    {'name': '面部几何', 'x': 9.6, 'bg': colors['orange'], 'dark': colors['orange_dark'],
     'steps': ['脸部位姿裁剪', '3D人脸重建', '几何系数推导\n(64维)', '面部几何嵌入']},
    {'name': '头部运动', 'x': 14.4, 'bg': colors['purple'], 'dark': colors['purple_dark'],
     'steps': ['全局关键点检测', '运动轨迹分析', '运动幅度计算\n(256维)', '动态趋势嵌入']},
    {'name': '面部表情', 'x': 19.2, 'bg': colors['red'], 'dark': colors['red_dark'],
     'steps': ['动作微元识别', '面部表情编码', '表情特征对齐\n(128维)', '深层表情嵌入']}
]

# 通用函数：带阴影的圆角矩形
def add_shadowed_box(ax, xy, width, height, facecolor, edgecolor, text=None, text_size=12, text_color='black', weight='normal', alpha=0.9, pad=0.1):
    shadow_pad = pad
    shadow = FancyBboxPatch((xy[0]+0.15, xy[1]-0.15), width, height, boxstyle=f'round,pad={shadow_pad}', 
                            facecolor=colors['shadow'], edgecolor='none', alpha=0.3, zorder=1)
    ax.add_patch(shadow)
    box = FancyBboxPatch(xy, width, height, boxstyle=f'round,pad={pad}', 
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + width/2, xy[1] + height/2, text, ha='center', va='center', 
                fontsize=text_size, color=text_color, fontweight=weight, zorder=3)

# ================= 1. 顶部输入流及分发总线 =================
# 顶部输入：替换为本地图片 1.png
if os.path.exists('1.png'):
    try:
        img = mpimg.imread('1.png')
        img_h, img_w = img.shape[:2]
        
        # 预设外框区域大小：宽 8.0，高 1.2
        box_w = 8.0
        box_h = 1.2
        center_x = 12.0
        center_y = 13.4
        
        # 保持比例缩放图片
        aspect_img = img_w / img_h
        aspect_box = box_w / box_h
        if aspect_img > aspect_box:
            target_w = box_w
            target_h = target_w / aspect_img
        else:
            target_h = box_h
            target_w = target_h * aspect_img
            
        left, right = center_x - target_w / 2, center_x + target_w / 2
        bottom, top = center_y - target_h / 2, center_y + target_h / 2
        
        # 绘制背景阴影框以保持一致的UI风格（不带文字）
        add_shadowed_box(ax, (center_x - box_w/2, center_y - box_h/2), box_w, box_h, colors['header'], '#999999', '', pad=0.15)
        # 叠加图片，并设置较高的zorder以防被遮挡
        ax.imshow(img, extent=[left, right, bottom, top], zorder=4)
    except Exception as e:
        print(f"加载图片失败，回退到文字框: {e}")
        add_shadowed_box(ax, (8.0, 13.0), 8.0, 0.8, colors['header'], '#999999', '原始全视域视频流', 16, weight='bold', pad=0.15)
else:
    # 顶部输入框
    add_shadowed_box(ax, (8.0, 13.0), 8.0, 0.8, colors['header'], '#999999', '原始全视域视频流', 16, weight='bold', pad=0.15)

# 输入向下的主管线
ax.plot([12.0, 12.0], [13.0-0.15, 12.4], color='#777777', lw=2.5, zorder=1)
# 水平分发总线
ax.plot([4.8, 19.2], [12.4, 12.4], color='#777777', lw=2.5, zorder=1)

# ================= 2. 并行五大模态提取区 =================
y_positions = [9.8, 8.0, 6.2, 4.4]

for pipe in pipelines:
    x = pipe['x']
    ax.annotate('', xy=(x, 11.6+0.05), xytext=(x, 12.4),
                arrowprops=dict(arrowstyle='->', color='#777777', lw=2.5), zorder=1)
    
    add_shadowed_box(ax, (x-1.8, 3.8), 3.6, 7.2, pipe['bg'], pipe['dark'], alpha=0.35, pad=0.15)
    
    bg_box = FancyBboxPatch((x-1.8, 11.0), 3.6, 0.6, boxstyle='round,pad=0.0', 
                            facecolor=pipe['dark'], edgecolor=pipe['dark'], zorder=2)
    ax.add_patch(bg_box)
    ax.text(x, 11.3, pipe['name'], ha='center', va='center', fontsize=13, color='white', fontweight='bold', zorder=3)
    
    for i, step in enumerate(pipe['steps']):
        y = y_positions[i]
        add_shadowed_box(ax, (x-1.4, y-0.35), 2.8, 0.7, '#FFFFFF', pipe['dark'], step, 11, pad=0.08)
        
        if i < len(pipe['steps']) - 1:
            next_y = y_positions[i+1]
            ax.annotate('', xy=(x, next_y+0.43+0.02), xytext=(x, y-0.43-0.02),
                        arrowprops=dict(arrowstyle='->', color=pipe['dark'], lw=2.5), zorder=1)
    
    ax.plot([x, x], [3.8-0.15, 3.0], color='#555555', lw=2.5, zorder=1)

# ================= 3. 底部特征融合及降维区 =================
# 底部汇聚水平总线
ax.plot([4.8, 19.2], [3.0, 3.0], color='#555555', lw=2.5, zorder=1)

# 总线流向融合层箭头
ax.annotate('', xy=(12.0, 2.2+0.15+0.02), xytext=(12.0, 3.0),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5), zorder=1)

# 模态特征拼接层
add_shadowed_box(ax, (7.0, 1.4), 10.0, 0.8, colors['middle'], '#AAB7B8', '全局模态特征混合拼接 (128+64+256+128) 维', 14, weight='bold', pad=0.15)

ax.annotate('', xy=(12.0, 0.8+0.15+0.02), xytext=(12.0, 1.4-0.15-0.02),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5), zorder=1)

# PCA 层
add_shadowed_box(ax, (8.5, 0.0), 7.0, 0.8, '#D1F2EB', '#76D7C4', 'PCA 主成分分析与降维映射', 14, weight='bold', pad=0.15)

ax.annotate('', xy=(12.0, -0.8+0.15+0.02), xytext=(12.0, 0.0-0.15-0.02),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5), zorder=1)

add_shadowed_box(ax, (5.0, -1.6), 14.0, 0.8, '#FADBD8', '#E74C3C', '【 最终融合表示特征向量 】', 15, weight='bold', pad=0.15)

plt.tight_layout()
plt.savefig('feature_fusion_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
