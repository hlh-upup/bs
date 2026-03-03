import re

new_bottom = """# Bottom Text
b_y = -0.4
b_line_y = 0.2

# Horizontal line connecting the losses
ax.plot([col_enc, col_fus], [b_line_y, b_line_y], '-', color='black', lw=1.5, zorder=5)

b_loss1 = "MSE + L1 混合损失及可学习Sigmoid权重 [cite: 3.2.3]"
ax.text((col_enc + col_att)/2, b_y, b_loss1, ha='center', va='center', fontsize=11, weight='bold')

b_loss2 = "跨任务一致性正则化损失 [cite: 3.2.3]"
ax.text((col_pool + col_fus)/2 + 0.5, b_y, b_loss2, ha='center', va='center', fontsize=11, weight='bold')

b_loss3 = "*. 针对缺失标签的标签掩码机制 [cite: 3.2.3]"
ax.text(30.5, b_y, b_loss3, ha='right', va='center', fontsize=10)

# Up arrows from bottom line to the modules
# Arrow to encoder
draw_arrow(ax, (col_enc, b_line_y), (col_enc, 1.4), lw=1.5)
# Arrow to attention
draw_arrow(ax, (col_att, b_line_y), (col_att, 1.4), lw=1.5)
# Arrow to fusion (Path 2 dashed box bottom)
draw_arrow(ax, (col_fus, b_line_y), (col_fus, 2.0), lw=1.5)

plt.tight_layout()"""

with open('test.py', 'r', encoding='utf-8') as f:
    orig = f.read()

orig = re.sub(r'# Bottom Text.*plt\.tight_layout\(\)', new_bottom, orig, flags=re.DOTALL)

with open('test.py', 'w', encoding='utf-8') as f:
    f.write(orig)
