#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import pandas as pd
from datetime import datetime
import gradio as gr

REQUIRED_COLUMNS = ["item_id", "video_a", "video_b"]
TASKS = ["lip_sync", "expression", "audio_quality", "cross_modal", "overall"]


def _read_pairs_csv(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {col}. 需要列: {REQUIRED_COLUMNS}")

    def _normalize_path(p: str) -> str:
        if isinstance(p, str) and p.lower().startswith(("http://", "https://")):
            return p.strip()
        # 本地路径转绝对路径，并校验存在性
        ap = os.path.abspath(str(p))
        if not os.path.exists(ap):
            # 不直接报错，允许先加载任务；在播放时由浏览器报错更直观
            # 也可改为 raise ValueError(f"本地文件不存在: {ap}")
            pass
        return ap

    df["video_a"] = df["video_a"].apply(_normalize_path)
    df["video_b"] = df["video_b"].apply(_normalize_path)
    return df


def _prepare_session(df: pd.DataFrame, shuffle: bool = True):
    records = df.to_dict(orient="records")
    if shuffle:
        random.shuffle(records)
    # 对每个条目随机左右位置（left/right 对应 A 或 B）
    order_list = []
    for r in records:
        if random.random() < 0.5:
            order_list.append({
                "item_id": r["item_id"],
                "left_label": "A", "right_label": "B",
                "left_path": r["video_a"], "right_path": r["video_b"],
            })
        else:
            order_list.append({
                "item_id": r["item_id"],
                "left_label": "B", "right_label": "A",
                "left_path": r["video_b"], "right_path": r["video_a"],
            })
    return order_list


def _append_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")


def build_interface():
    with gr.Blocks(title="A/B 视频主观评测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        #视频主观评
        - 上传配对 CSV（列：`item_id, video_a, video_b`）
        - 系统会对每个条目随机左右顺序
        - 对左右视频分别在 5 个维度打分（1~5），点击“提交 & 下一条”
        - 评分会即时追加保存到 CSV，可点击右侧按钮下载
        """)

        with gr.Row():
            csv_file = gr.File(label="上传配对 CSV", file_count="single", file_types=[".csv"])
            rater_id = gr.Textbox(label="评测者ID(必填)", placeholder="如: rater_001")
            out_csv = gr.Textbox(label="输出CSV路径", value="ab_eval_results.csv")
            load_btn = gr.Button("加载任务")

        status = gr.Markdown()

        with gr.Row():
            with gr.Column():
                # 固定尺寸，避免加载不同分辨率视频时组件突变
                video_left = gr.Video(label="左侧视频", width=480, height=360)
            with gr.Column():
                video_right = gr.Video(label="右侧视频", width=480, height=360)

        with gr.Row():
            with gr.Column():
                left_scores = [
                    gr.Slider(1, 5, step=0.1, value=3.0, label=f"左-{t} (1-5，0.1步长)") for t in TASKS
                ]
            with gr.Column():
                right_scores = [
                    gr.Slider(1, 5, step=0.1, value=3.0, label=f"右-{t} (1-5，0.1步长)") for t in TASKS
                ]

        comments = gr.Textbox(label="可选备注", placeholder="主观感受或异常情况记录…")

        with gr.Row():
            submit_btn = gr.Button("提交 & 下一条", variant="primary")
            download_btn = gr.DownloadButton("下载当前CSV", variant="secondary")
            progress_md = gr.Markdown()

        # 状态
        order_state = gr.State([])  # list of dict for each item
        idx_state = gr.State(0)
        total_state = gr.State(0)
        csv_path_state = gr.State("")

        def on_load(csv_f, out_path, rid):
            if csv_f is None:
                return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                        [], 0, 0, "", gr.update(visible=False), "请先上传 CSV")
            try:
                df = _read_pairs_csv(csv_f.name)
                order_list = _prepare_session(df, shuffle=True)
                total = len(order_list)
                if total == 0:
                    return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                            [], 0, 0, "", gr.update(visible=False), "CSV 无有效条目")
                # 设置下载按钮目标
                download_btn_file = os.path.abspath(out_path or "ab_eval_results.csv")
                # 初始化首条
                first = order_list[0]
                left_v = first["left_path"]
                right_v = first["right_path"]
                msg = f"已加载 {total} 条目。评测者: {rid or '(未填)'}"
                return (left_v, right_v, f"进行中：1/{total}", order_list, 0, total, download_btn_file, gr.update(value=download_btn_file, visible=True), msg)
            except Exception as e:
                return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                        [], 0, 0, "", gr.update(visible=False), f"加载失败: {e}")

        load_btn.click(
            on_load,
            inputs=[csv_file, out_csv, rater_id],
            outputs=[video_left, video_right, progress_md, order_state, idx_state, total_state, csv_path_state, download_btn, status]
        )

        def on_submit(rid, out_path, order_list, idx, total,
                      *scores_and_comment):
            # 帮助函数：生成保持滑块不变的占位更新，确保返回数量一致
            def _no_change_sliders():
                return [gr.update()] * (len(TASKS) * 2)

            if not rid:
                return (gr.update(), gr.update(), gr.update(), gr.update(), "请先填写评测者ID", *_no_change_sliders())
            if not order_list or idx >= total:
                return (gr.update(), gr.update(), gr.update(), gr.update(), "没有可评测的条目，请先加载任务", *_no_change_sliders())

            # 拆解分数
            # scores_and_comment = [L_lipsync, L_expression, ..., L_overall, R_lipsync, ..., R_overall, comments]
            if len(scores_and_comment) != len(TASKS)*2 + 1:
                return (gr.update(), gr.update(), gr.update(), gr.update(), "内部参数不一致，请刷新页面重试", *_no_change_sliders())
            left_vals = scores_and_comment[:len(TASKS)]
            right_vals = scores_and_comment[len(TASKS):len(TASKS)*2]
            cmts = scores_and_comment[-1]

            current = order_list[idx]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            row = {
                "timestamp": now,
                "rater_id": rid,
                "item_id": current["item_id"],
                "left_label": current["left_label"],
                "right_label": current["right_label"],
                "left_path": current["left_path"],
                "right_path": current["right_path"],
            }
            for i, t in enumerate(TASKS):
                try:
                    lv = round(float(left_vals[i]), 1)
                except Exception:
                    lv = 3.0
                try:
                    rv = round(float(right_vals[i]), 1)
                except Exception:
                    rv = 3.0
                row[f"left_{t}"] = lv
                row[f"right_{t}"] = rv
            row["comments"] = cmts or ""

            # 追加写入 CSV
            csv_abs = os.path.abspath(out_path or "ab_eval_results.csv")
            try:
                _append_row(csv_abs, row)
            except Exception as e:
                return (gr.update(), gr.update(), gr.update(), gr.update(), f"写入失败: {e}", *_no_change_sliders())

            # 下一条
            idx += 1
            if idx >= total:
                # 评测完成，视频清空，保持滑块不变，并弹出提示
                try:
                    gr.Info("评测完成，感谢参与！")
                except Exception:
                    pass
                return (gr.update(value=None), gr.update(value=None), f"已完成：{total}/{total}", idx, "评测完成，感谢参与！", *_no_change_sliders())

            nxt = order_list[idx]
            left_v = nxt["left_path"]
            right_v = nxt["right_path"]
            prog = f"进行中：{idx+1}/{total}"

            # 清空打分（重置为默认 3.0 分）
            reset_updates = [gr.update(value=3.0)] * (len(TASKS)*2)

            return (left_v, right_v, prog, idx, "已保存，进入下一条", *reset_updates)

        # 注意：outputs 数量需严格匹配
        submit_btn.click(
            on_submit,
            inputs=[rater_id, out_csv, order_state, idx_state, total_state, *left_scores, *right_scores, comments],
            outputs=[video_left, video_right, progress_md, idx_state, status, *left_scores, *right_scores]
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # 0.0.0.0 便于局域网访问；Windows 默认 7860 端口
    # 允许访问当前工作目录与当前盘符根目录下的本地文件（如 F:\zhipu_test\...），否则 Gradio 会因安全策略拒绝缓存/读取
    try:
        drive_root = os.path.splitdrive(os.getcwd())[0] + os.sep  # 例如 "F:\\"
        allowed_dirs = [os.getcwd(), drive_root]
    except Exception:
        allowed_dirs = [os.getcwd()]
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        allowed_paths=allowed_dirs,
    )
