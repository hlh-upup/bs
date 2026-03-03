import re

with open('test.py', 'r', encoding='utf-8') as f:
    orig = f.read()

replacement = """# Average and Global Vector
draw_box(ax, col_avg, 7.0, 1.2, 0.8, "均值融合", 'white', lw=1.5)

# 修复路径1到均值融合的连接 (21.1是Mean-pool绿框的右边缘)
draw_arrow(ax, (21.1, y_p1_4), (col_avg - 0.6, 7.2), conn="arc3,rad=-0.1")
# 修复路径2到均值融合的连接 (21.6是多尺度融合框的右边缘)
draw_arrow(ax, (21.6, p2_y_center), (col_avg - 0.6, 6.8), conn="arc3,rad=0.1")

ax.text(21.75, 7.3, "512", ha='center', va='bottom', fontsize=9)
ax.text(22.0, 4.8, "512", ha='center', va='top', fontsize=9)"""

# I need to match everything from # Draw bus connection to Average up to the Global Fusion Vector
old_regex = r'# Draw bus connection to Average.*?(?=draw_box\(ax, col_gfv)'
orig = re.sub(old_regex, replacement + '\n', orig, flags=re.DOTALL)

with open('test.py', 'w', encoding='utf-8') as f:
    f.write(orig)
print("Updated!")
