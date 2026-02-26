# -*- coding: utf-8 -*-
import os
import sys
import argparse
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

CANDIDATES = [
    "Microsoft YaHei",  # 微软雅黑（Windows首选）
    "SimHei",           # 黑体
    "SimSun",           # 宋体
    "DengXian",         # 等线
    "Noto Sans CJK SC", # 思源黑体
    "Source Han Sans SC",
    "Noto Serif CJK SC",
    "Arial Unicode MS",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",      # 兜底
]

TEST_TEXT = "总体评分｜预测｜真实｜差值｜Bland–Altman 图"

def pick_font():
    installed = {f.name for f in fm.fontManager.ttflist}
    available = [f for f in CANDIDATES if f in installed]
    return available, (available[0] if available else None)

def make_test_plot(save_path, font_family):
    mpl.rcParams["font.sans-serif"] = [font_family]
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 4))
    plt.title("字体测试 Font Check")
    plt.text(0.05, 0.6, f"已选字体: {font_family}", fontsize=12)
    plt.text(0.05, 0.35, TEST_TEXT, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/font_test.png", help="测试图输出路径")
    args = ap.parse_args()

    available, chosen = pick_font()
    print("候选中文字体优先级：", CANDIDATES)
    print("本机可用字体：", available)
    if not chosen:
        print("未找到可用中文字体，建议安装：Microsoft YaHei 或 Noto Sans CJK SC。")
        sys.exit(1)

    print("推荐字体：", chosen)
    try:
        make_test_plot(args.out, chosen)
        print("已生成测试图：", args.out)
        print("在 Matplotlib 中使用：")
        print(f"  import matplotlib as mpl\n  mpl.rcParams['font.sans-serif'] = ['{chosen}']\n  mpl.rcParams['axes.unicode_minus'] = False")
    except Exception as e:
        print("生成测试图失败：", e)
        sys.exit(2)

if __name__ == "__main__":
    main()