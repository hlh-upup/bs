import re

with open('test.py', 'r', encoding='utf-8') as f:
    orig = f.read()

old_code = """draw_arrow(ax, (col_fus + 1.9, y_p1_4), (col_avg - 0.6, 7.2), conn="arc3,rad=-0.1")
draw_arrow(ax, (col_fus + 1.9, p2_y_center), (col_avg - 0.6, 6.8), conn="arc3,rad=0.1")"""

new_code = """# Draw bus connection to Average
out_bus_x = col_avg - 1.2
draw_arrow(ax, (col_fus + 1.9, y_p1_4), (out_bus_x, y_p1_4), style='-')
draw_arrow(ax, (col_fus + 1.9, p2_y_center), (out_bus_x, p2_y_center), style='-')
# Vertical line linking path 1 and path 2
ax.plot([out_bus_x, out_bus_x], [p2_y_center, y_p1_4], '-', color='black', lw=1.5, zorder=5)
# Arrow entering Average
draw_arrow(ax, (out_bus_x, 7.0), (col_avg - 0.6, 7.0))"""

if old_code in orig:
    orig = orig.replace(old_code, new_code)
    with open('test.py', 'w', encoding='utf-8') as f:
        f.write(orig)
    print('Updated path 2 connections successfully.')
else:
    print('Could not find old code')