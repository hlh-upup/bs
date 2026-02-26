#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从本地视频目录生成 A/B 配对 CSV

用法示例：
  python tools\generate_ab_pairs.py --video_dir F:\bs\datasets\ch-simsv2s\Raw\aqgy3_0001 --output apps\local_pairs.csv --limit 6 --shuffle

输出 CSV 列：item_id, video_a, video_b
"""
import os
import sys
import argparse
import csv
import random
from pathlib import Path

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def find_videos(video_dir: Path, recursive: bool = False):
    if recursive:
        files = [p for p in video_dir.rglob('*') if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    else:
        files = [p for p in video_dir.glob('*') if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    # 去重并排序
    uniq = sorted({str(p.resolve()) for p in files})
    return uniq


def make_pairs(paths, limit: int = None, shuffle: bool = False):
    items = list(paths)
    if shuffle:
        random.shuffle(items)
    # 将相邻文件两两成对：(0,1), (2,3), ...
    pairs = []
    for i in range(0, len(items) - 1, 2):
        pairs.append((items[i], items[i+1]))
    if limit is not None and limit > 0:
        pairs = pairs[:limit]
    return pairs


def main():
    parser = argparse.ArgumentParser(description='生成 A/B 配对 CSV')
    parser.add_argument('--video_dir', required=True, help='视频目录（本地路径）')
    parser.add_argument('--output', required=True, help='输出 CSV 路径')
    parser.add_argument('--limit', type=int, default=6, help='最多生成多少条配对（默认6）')
    parser.add_argument('--shuffle', action='store_true', help='是否打乱视频列表后再配对')
    parser.add_argument('--recursive', action='store_true', help='是否递归扫描子目录')
    parser.add_argument('--relative_to', type=str, default=None, help='将输出的 video 路径转换为相对于该目录的相对路径（推荐设置为 Gradio 允许的根目录例如项目根）')
    parser.add_argument('--make_relative_to_cwd', action='store_true', help='快捷选项：等同于 --relative_to 当前工作目录')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists() or not video_dir.is_dir():
        print(f'[Error] 视频目录不存在或不是文件夹: {video_dir}', file=sys.stderr)
        sys.exit(1)

    videos = find_videos(video_dir, recursive=args.recursive)
    if len(videos) < 2:
        print(f'[Error] 目录中可用视频不足 2 个: {video_dir}', file=sys.stderr)
        sys.exit(2)

    pairs = make_pairs(videos, limit=args.limit, shuffle=args.shuffle)
    if not pairs:
        print('[Error] 未生成任何配对，请检查 limit 或目录内容', file=sys.stderr)
        sys.exit(3)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 路径相对化（可选）
    rel_base: Path | None = None
    if args.make_relative_to_cwd:
        rel_base = Path.cwd()
    elif args.relative_to:
        rb = Path(args.relative_to)
        if rb.exists() and rb.is_dir():
            rel_base = rb.resolve()
        else:
            print(f'[Warn] --relative_to 指定的目录不存在或不可用: {rb}，将使用绝对路径', file=sys.stderr)

    def to_rel(p: str) -> str:
        if rel_base is None:
            return p
        try:
            return str(Path(p).resolve().relative_to(rel_base))
        except Exception:
            # 无法相对化则返回原始绝对路径
            return p

    with out_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'video_a', 'video_b'])
        for idx, (a, b) in enumerate(pairs, start=1):
            item_id = f'{idx:03d}'
            writer.writerow([item_id, to_rel(a), to_rel(b)])

    print(f'[OK] 已生成 {len(pairs)} 条配对 -> {out_path.resolve()}')


if __name__ == '__main__':
    main()
