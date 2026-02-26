# -*- coding: utf-8 -*-
"""
列出系统字体并检测是否包含指定中文字符集合。
- 默认检测文本包含：总体评分 预测 真实 差值 Bland-Altman 图
- 输出：控制台 + CSV + Markdown，可选生成测试图验证显示效果
使用示例（Windows 命令行）：
  python tools\list_fonts_and_check_glyphs.py ^
    --check-text "总体评分 预测 真实 差值 Bland-Altman 图" ^
    --out-csv reports\fonts_report.csv ^
    --out-md reports\fonts_report.md ^
    --test-plot reports\fonts_test.png
"""
import argparse
import os
import sys
from collections import defaultdict

# 优先尽量使用 matplotlib 自带的字体扫描
from matplotlib import font_manager as fm
import matplotlib

# 尝试使用 fontTools 更准确检测字形覆盖（推荐）
try:
    from fontTools.ttLib import TTFont
    HAS_FONTTOOLS = True
except Exception:
    HAS_FONTTOOLS = False

PREFERRED_FAMILIES = [
    "Microsoft YaHei",      # 微软雅黑
    "SimHei",               # 黑体
    "SimSun",               # 宋体
    "Noto Sans CJK SC",     # 思源黑体 SC
    "Noto Serif CJK SC",    # 思源宋体 SC
    "Source Han Sans SC",   # 同上别名
    "Source Han Serif SC",
    "Arial Unicode MS",     # 全字库
    "PingFang SC",          # 苹方（macOS）
    "Heiti SC",
    "WenQuanYi Zen Hei",    # 文泉驿
    "DejaVu Sans",          # 兜底（覆盖少量常用汉字）
]

def list_system_font_files():
    # 扫描 ttf/otf/ttc
    fonts = set()
    for ext in ("ttf", "otf", "ttc"):
        try:
            fonts.update(fm.findSystemFonts(fontext=ext))
        except Exception:
            pass
    return sorted(fonts)

def get_font_family_name(font_path, font_number=None):
    # 用 matplotlib 解析家族名（对 ttc 可能不准确）
    try:
        if font_number is None:
            prop = fm.FontProperties(fname=font_path)
        else:
            prop = fm.FontProperties(fname=f"{font_path}#{font_number}")
        name = prop.get_name()
        return name
    except Exception:
        return None

def _load_cmap_codepoints_ttf(font_path):
    cps = set()
    try:
        font = TTFont(font_path, fontNumber=None)
        for table in font["cmap"].tables:
            if table.isUnicode():
                cps.update(table.cmap.keys())
        font.close()
    except Exception:
        pass
    return cps

def _load_cmap_codepoints_ttc(font_path, max_faces=8):
    faces = []
    for idx in range(max_faces):
        try:
            font = TTFont(font_path, fontNumber=idx)
            cps = set()
            for table in font["cmap"].tables:
                if table.isUnicode():
                    cps.update(table.cmap.keys())
            name = get_font_family_name(font_path, font_number=idx) or f"{os.path.basename(font_path)}#{idx}"
            faces.append((name, cps, idx))
            font.close()
        except Exception:
            break
    return faces

def font_supports_text(font_path, text):
    """返回 (family_name, coverage_ratio, missing_chars, face_index)"""
    family = get_font_family_name(font_path)
    if not text:
        return family, 1.0, [], None
    chars = [c for c in text if not c.isspace()]
    uniq = sorted(set(chars))
    if HAS_FONTTOOLS:
        if font_path.lower().endswith(".ttc"):
            faces = _load_cmap_codepoints_ttc(font_path)
            best = None
            for name, cps, face_idx in faces:
                covered = [c for c in uniq if ord(c) in cps]
                ratio = len(covered) / max(1, len(uniq))
                missing = [c for c in uniq if ord(c) not in cps]
                if best is None or ratio > best[1]:
                    best = (name or family, ratio, missing, face_idx)
            if best is None:
                return family, 0.0, uniq, None
            return best
        else:
            cps = _load_cmap_codepoints_ttf(font_path)
            covered = [c for c in uniq if ord(c) in cps]
            ratio = len(covered) / max(1, len(uniq))
            missing = [c for c in uniq if ord(c) not in cps]
            return family, ratio, missing, None
    else:
        # 无 fontTools：退化为粗略判断（尝试设置 rcParams 并渲染文本长度）
        # 仅返回 family 和 0/1 覆盖（不准确，但可做初筛）
        family = family or os.path.splitext(os.path.basename(font_path))[0]
        try:
            matplotlib.rcParams["font.sans-serif"] = [family]
            matplotlib.rcParams["axes.unicode_minus"] = False
            # 粗略假设支持
            return family, 0.5, [], None
        except Exception:
            return family, 0.0, uniq, None

def choose_best_font(candidates):
    """按优先名单挑选最优中文字体"""
    # candidates: list of dict with keys: family, ratio, path, face_index
    fam_to_best = defaultdict(lambda: (-1.0, None))
    for c in candidates:
        r, _ = fam_to_best[c["family"]]
        if c["ratio"] > r:
            fam_to_best[c["family"]] = (c["ratio"], c)
    # 先按优先名单，再按覆盖率排序
    ordered = []
    used = set()
    for fam in PREFERRED_FAMILIES:
        if fam in fam_to_best:
            ordered.append(fam_to_best[fam][1])
            used.add(fam)
    for fam, (_, c) in sorted(fam_to_best.items(), key=lambda kv: kv[1][0], reverse=True):
        if fam not in used:
            ordered.append(c)
    return ordered

def make_test_plot(save_path, font_family, text):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [font_family]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 4))
    plt.title("字体测试 Font Check")
    plt.text(0.05, 0.7, f"已选字体: {font_family}", fontsize=12)
    plt.text(0.05, 0.45, f"测试文本: {text}", fontsize=14)
    plt.text(0.05, 0.2, "Bland–Altman 图 / 总体评分 / 预测 真实 差值", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="列出系统字体并检测是否包含指定中文字符集合")
    parser.add_argument("--check-text", type=str,
                        default="总体评分 预测 真实 差值 Bland-Altman 图",
                        help="用于检测字形覆盖的文本（建议包含你图表里用到的所有中文）")
    parser.add_argument("--out-csv", type=str, default="reports/fonts_report.csv", help="CSV 输出路径")
    parser.add_argument("--out-md", type=str, default="reports/fonts_report.md", help="Markdown 输出路径")
    parser.add_argument("--test-plot", type=str, default="", help="可选：用选中字体生成测试图 PNG 路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    font_files = list_system_font_files()
    if not font_files:
        print("未找到系统字体。")
        sys.exit(1)

    if not HAS_FONTTOOLS:
        print("警告：未安装 fontTools，将使用粗略检测。建议安装：pip install fonttools")

    records = []
    for fp in font_files:
        family, ratio, missing, face_idx = font_supports_text(fp, args.check_text)
        records.append({
            "family": family or "",
            "path": fp,
            "face_index": face_idx if face_idx is not None else "",
            "coverage_ratio": round(ratio, 4),
            "missing_chars": "".join(missing)
        })

    # 选择最佳字体列表
    ordered = choose_best_font(records)
    best = next((r for r in ordered if r["coverage_ratio"] >= 1.0), None)
    best_or_top = best or (ordered[0] if ordered else None)

    # 控制台输出 Top 10
    print("\n检测文本：", args.check_text)
    print("\nTop 候选字体（按优先名单与覆盖率综合排序，取前 10）：")
    for i, r in enumerate(ordered[:10], 1):
        print(f"{i:>2}. {r['family']}  覆盖率={r['coverage_ratio']:.2f}  {'(完全覆盖)' if r['coverage_ratio']==1.0 else ''}")

    if best_or_top:
        print("\n推荐使用字体：", best_or_top["family"])
        print("Matplotlib 设置示例：")
        print("  import matplotlib as mpl")
        print(f"  mpl.rcParams['font.sans-serif'] = ['{best_or_top['family']}']")
        print("  mpl.rcParams['axes.unicode_minus'] = False")

    # 写 CSV
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "coverage_ratio", "face_index", "path", "missing_chars"])
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x["coverage_ratio"], x["family"]), reverse=True):
            writer.writerow(r)

    # 写 MD
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# 字体检测报告\n\n")
        f.write(f"- 检测文本：{args.check_text}\n")
        f.write(f"- 字体总数：{len(records)}\n")
        f.write(f"- 已安装 fontTools：{HAS_FONTTOOLS}\n\n")
        f.write("## 推荐字体\n\n")
        if best_or_top:
            f.write(f"- {best_or_top['family']}（覆盖率 {best_or_top['coverage_ratio']:.2f}）\n\n")
            f.write("Matplotlib 设置示例：\n")
            f.write("```python\n")
            f.write("import matplotlib as mpl\n")
            f.write(f"mpl.rcParams['font.sans-serif'] = ['{best_or_top['family']}']\n")
            f.write("mpl.rcParams['axes.unicode_minus'] = False\n")
            f.write("```\n\n")
        else:
            f.write("- 未找到合适字体\n\n")
        f.write("## Top 10 候选\n\n")
        for i, r in enumerate(ordered[:10], 1):
            tag = "（完全覆盖）" if r["coverage_ratio"] == 1.0 else ""
            f.write(f"{i}. {r['family']} - 覆盖率 {r['coverage_ratio']:.2f}{tag}\n")
        f.write("\n> 完整列表见 CSV。\n")

    # 生成测试图
    if args.test_plot and best_or_top:
        try:
            os.makedirs(os.path.dirname(args.test_plot), exist_ok=True)
        except Exception:
            pass
        make_test_plot(args.test_plot, best_or_top["family"], args.check_text)
        print("已生成测试图：", args.test_plot)

if __name__ == "__main__":
    main()