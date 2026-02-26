import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 与 evaluator.py 保持一致的字体与主题设置（优先使用微软雅黑）
import matplotlib as _mpl
_mpl.rcParams['font.family'] = 'sans-serif'
_mpl.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 优先微软雅黑
    'SimHei',            # 黑体
    'Arial Unicode MS',  # 全字库
    'Noto Sans CJK SC',  # Noto CJK
    'DejaVu Sans',       # 兜底
]
_mpl.rcParams['axes.unicode_minus'] = False
sns.set_theme(
    style='whitegrid',
    rc={
        'font.family': 'sans-serif',
        'font.sans-serif': _mpl.rcParams['font.sans-serif'],
        'axes.unicode_minus': False,
    }
)

# 数据
labels = ["口型同步", "表情自然度", "音频质量", "跨模态一致性", "总体评分"]
values = [0.554, 0.437, 0.621, 0.596, 0.698]

# 绘制柱状图（保持 evaluator 中的风格：不强制颜色，使用主题默认）
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, label='柱形图')

# Pearson 柱状图与 evaluator.py 保持一致：仅柱形与数值标注，不叠加折线

# 在柱顶显示数值（与 evaluator 一致的格式）
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
             ha='center', va='bottom', rotation=0)

# 标题与坐标轴标签（中文，与 evaluator 一致）
plt.title("各任务的皮尔逊相关")
plt.xlabel("任务")
plt.ylabel("皮尔逊相关")

# evaluator.py 的参考线通常不加图例，这里保持一致不显示图例

# 保存图像（与 evaluator 一致：tight_layout 后保存）
plt.tight_layout()
plt.savefig("pearson_with_lipsync_final.png", dpi=300, bbox_inches='tight')
plt.close()
