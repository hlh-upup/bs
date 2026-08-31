# 图3-2 与图3-3 架构图 Gemini 生图 Prompt

## 图3-2 多模态特征提取模块架构图

Generate a clean, professional technical architecture diagram for an academic paper (Chinese labels). White background, flat design style, clear structure with rounded-corner rectangles and directional arrows.

Layout: top-to-bottom flow, divided into two main sections by a horizontal dashed line.

**Top section — "并行特征提取层" (Parallel Feature Extraction Layer):**

At the very top center, place a single rounded rectangle labeled "输入视频 (150帧, 25fps)" in blue fill. From this box, draw five downward arrows fanning out to five parallel vertical processing pipelines, each in a different color:

Pipeline 1 (green): A vertical chain of rounded boxes connected by downward arrows:
"视频帧" → "RetinaFace 人脸检测" → "224×224 裁剪 + ImageNet归一化" → "ResNet101 (去FC层)" → "GAP" → a box labeled "2048维/帧" at the bottom.

Pipeline 2 (orange): A vertical chain:
"FFmpeg 分离音频" → "16kHz PCM" → "HuBERT-Base" → "768维" → "线性插值对齐" → a box labeled "768维/帧".

Pipeline 3 (purple): A vertical chain:
"MediaPipe Face Mesh" → "468个3D关键点" → "展平" → a box labeled "1404维/帧".

Pipeline 4 (red): A vertical chain:
"关键点几何距离" → "17个AU强度" → "最大值归一化" → a box labeled "17维/帧". Draw a small dashed arrow from Pipeline 3's "468个3D关键点" box sideways into this pipeline's first box, showing the dependency.

Pipeline 5 (gray, slightly separated to the right with a dashed border): A vertical chain:
"SyncNet" → "同步置信度 + 偏移量" → "Sigmoid映射" → a box labeled "[0,5]分数 (视频级)". Add a small annotation "(辅助参考，不参与时序建模)".

Below pipelines 1–4, merge their bottom output boxes with four converging arrows into a single wide box labeled "原始特征拼接: 4237维 (2048+768+1404+17)".

**Bottom section — "特征预处理与降维层" (Feature Preprocessing & Dimensionality Reduction Layer):**

Below the dashed dividing line, draw a left-to-right horizontal processing chain of three rounded boxes connected by arrows:
Box 1 (light yellow): "NaN中位数插补"
Box 2 (light blue): "Z-score标准化"
Box 3 (light green): "PCA降维"

Next to Box 3, place a small table or annotation box showing the per-modality PCA configuration:
"视觉: 2048→100维 (≥95%方差)"
"音频: 768→200维 (≥95%方差)"
"关键点: 1404→50维 (≥85%方差)"
"AU: 17维 (不降维)"

From Box 3, draw a final downward arrow into a prominent output box at the very bottom, labeled "标准化特征输出: 367维 (100+200+50+17)", with a subtitle "→ 送入残差编码与跨模态融合模块".

Overall style: crisp vector-style lines, soft pastel fill colors, Chinese text labels in black, consistent font size, no 3D effects, suitable for insertion into an academic thesis.

---

## 图3-3 跨模态融合编码与多任务预测模块架构图

Generate a clean, professional technical architecture diagram for an academic paper (Chinese labels). White background, flat design style, left-to-right flow with rounded-corner rectangles and directional arrows. Landscape orientation.

Layout: left-to-right, four stages separated by thin dashed vertical lines.

**Stage 1 — "残差特征编码" (Residual Feature Encoding):**

Four color-coded input boxes stacked vertically on the left:
- Green: "视觉 (150×100)"
- Orange: "音频 (150×200)"
- Purple: "关键点 (150×50)"
- Red: "AU (150×17)"

Each → encoder block (rounded dashed border) containing:
- "Linear→LayerNorm→ReLU→Dropout" × 3 layers (stacked vertically)
- A curved dashed "残差捷径" arrow from input to output
- Visual path dimension example: "100→2048→1024→512"
Each encoder outputs "150×512".

**Stage 2 — "时序注意力 + 双池化" (Temporal Attention + Dual Pooling):**

All four 150×512 outputs → shared block "共享时序注意力 (8头MHSA + FFN, Dropout=0.1)".
After attention, each modality branch splits into "Mean Pool" and "Max Pool" → "Concat (1024维)" → "Linear→512维".
Result: four 512-dim vectors in respective colors.

**Stage 3 — "双路融合" (Dual-Path Fusion):**

Two parallel paths receiving four 512-dim vectors:
- Path A (light blue): "Transformer路径" — Stack [4×512] → Transformer编码器 (MHSA+FFN ×L层) → 均值池化 → z_tf
- Path B (light green): "多尺度路径" — 投影→拼接(2048维) → FC(1024)→ReLU→FC(512) → z_ms

Two outputs merge: "z = (z_tf + z_ms) / 2 (512维)"

**Stage 4 — "多任务预测" (Multi-Task Prediction):**

From z, five arrows → five prediction heads (each: "α缩放→Linear(256)→LN→GELU→Linear(128)→LN→GELU→Linear(1)→Sigmoid→[1,5]"):
- Blue: 唇形同步, Green: 表情自然度, Orange: 音频质量, Purple: 跨模态一致性, Red: 整体感知质量

**Bottom band:** Light gray, three boxes: "MSE+0.1×L1" | "w_k=σ(ω_k)" | "标签掩码+一致性正则"

Style: crisp vector lines, soft pastel fills, Chinese labels, no 3D effects, thesis-suitable.
