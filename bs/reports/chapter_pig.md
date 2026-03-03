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

## 图3-3 残差编码、跨模态双路融合与多任务预测模块架构图

Generate a clean, professional technical architecture diagram for an academic paper (Chinese labels). White background, flat design style, clear left-to-right flow layout with rounded-corner rectangles and directional arrows.

Layout: left-to-right flow, divided into five vertical stages separated by thin dashed vertical lines.

**Stage 1 (leftmost) — "三层残差特征编码器" (3-Layer Residual Feature Encoder):**

On the far left, place four horizontally aligned input boxes in different colors, stacked vertically:
- Green box: "视觉特征 (150×100)"
- Orange box: "音频特征 (150×200)"
- Purple box: "关键点特征 (150×50)"
- Red box: "AU特征 (150×17)"

Each box has a rightward arrow pointing to its own encoder block. Each encoder block is shown inside a rounded dashed border labeled "ImprovedFeatureEncoder", containing a vertical chain of three processing stages and a residual shortcut arrow:
- Inside each encoder block, show three identical stages stacked vertically: "Linear → LayerNorm → ReLU → Dropout(0.3)" × 3
- A curved dashed "残差捷径" (residual shortcut) arrow bypasses the three stages from input to output
- Dimension annotation for the green (visual) path example: "100 → 2048 → 1024 → 512"

Each encoder block outputs a box labeled "150×512维".

**Stage 2 — "时序自注意力 + 双池化" (Temporal Self-Attention + Dual Pooling):**

All four "150×512维" outputs flow rightward into a shared module block labeled "共享权重时序注意力 (Shared TemporalAttention)". Inside this block, show the Transformer-style structure:
- "多头自注意力 (8头)" with a residual skip-connection arrow + "LayerNorm"
- "前馈网络 FFN (dim×4)" with a residual skip-connection arrow + "LayerNorm"
- Mark "Dropout=0.1" as annotation.

After the temporal attention, show four parallel branches. Each branch shows a "双池化" (Dual Pooling) operation:
- Two parallel paths: "Mean Pool" and "Max Pool", converging with a "Concat" operation into "2×512=1024维"
- Followed by "Linear(1024, 512)" projection → output box "512维"

The result is four "512维" vector boxes (one per modality), each in its respective color (green/orange/purple/red).

**Stage 3 — "跨模态双路融合" (Cross-Modal Dual-Path Fusion):**

Show two parallel fusion paths, both receiving the four "512维" vectors as input:

**Path A (top, light blue background):** Labeled "Transformer跨模态融合路径":
- Four vectors stack into a matrix "[4×512]"
- Add "可学习位置编码 P ∈ R^{4×512}"
- Pass through a "多层Transformer编码器" block (show internal: "MHSA + FFN + Residual + LayerNorm" × L layers)
- "模态维度均值池化" → output: "z_tf (512维)"

**Path B (bottom, light green background):** Labeled "多尺度融合路径 (MultiScaleFusion)":
- Four vectors each go through "独立投影 Linear(512,512)"
- Concatenation: "[4×512] = 2048维"
- "FC(2048, 1024) → ReLU → Dropout(0.2) → FC(1024, 512)" → output: "z_ms (512维)"

The two path outputs converge with a "+" symbol and a "÷2" annotation, producing a single box labeled "融合向量 z = (z_tf + z_ms) / 2 (512维)".

**Stage 4 (rightmost) — "改进型多任务预测头" (Improved Multi-Task Prediction Heads):**

From the "融合向量 z (512维)" box, fan out five rightward arrows, each leading to an independent prediction head. The five heads are vertically stacked, each shown as a horizontal chain of small boxes in the same color. Each head starts with a small "α" symbol labeled "特征缩放" (feature scale), followed by:
- "Linear(512, 256) → LayerNorm → GELU → Dropout(0.3) → Linear(256, 128) → LayerNorm → GELU → Dropout(0.15) → Linear(128, 1) → Sigmoid → [1.0, 5.0]"

Final output labels for each head:
- Head 1 (blue): "唇形同步 (lip_sync)"
- Head 2 (green): "表情自然度 (expression)"
- Head 3 (orange): "音频质量 (audio_quality)"
- Head 4 (purple): "跨模态一致性 (cross_modal)"
- Head 5 (red): "整体感知质量 (overall)"

Each head ends with a small output circle showing a score value like "3.8" as an example.

**Bottom annotation area — "训练优化机制" (Training Optimization Mechanism):**

At the very bottom of the diagram, spanning the full width, place a light gray background band containing three annotation boxes:
- Left box (light yellow border): "混合损失函数" with items: "MSE + 0.1×L1"
- Center box (light orange border): "可学习Sigmoid权重" with formula: "w_k = σ(ω_k), L_total = Σ w_k · L_k + 0.1 · L_cons"
- Right box (light blue border): "标签掩码 + 一致性正则" with note: "无效标签(-1.0)掩码排除 | 跨任务std正则化(权重0.1)"

Overall style: crisp vector-style lines, soft pastel fill colors for each modality (consistent color coding throughout), Chinese text labels in black, consistent font size, no 3D effects, suitable for insertion into an academic thesis. The diagram should be wide (landscape orientation) to accommodate the left-to-right five-stage layout.
