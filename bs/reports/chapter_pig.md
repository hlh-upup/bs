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

From Box 3, draw a final downward arrow into a prominent output box at the very bottom, labeled "标准化特征输出: 367维 (100+200+50+17)", with a subtitle "→ 送入跨模态Transformer融合编码模块".

Overall style: crisp vector-style lines, soft pastel fill colors, Chinese text labels in black, consistent font size, no 3D effects, suitable for insertion into an academic thesis.

---

## 图3-3 多模态融合Transformer时序编码与多任务预测模块架构图

Generate a clean, professional technical architecture diagram for an academic paper (Chinese labels). White background, flat design style, clear left-to-right flow layout with rounded-corner rectangles and directional arrows.

Layout: left-to-right flow, divided into four vertical stages separated by thin dashed vertical lines.

**Stage 1 (leftmost) — "模态特征嵌入层" (Modal Feature Embedding Layer):**

On the far left, place four horizontally aligned input boxes in different colors, stacked vertically:
- Green box: "视觉特征 (150×100)"
- Orange box: "音频特征 (150×200)"
- Purple box: "关键点特征 (150×50)"
- Red box: "AU特征 (150×17)"

Each box has a rightward arrow pointing to its own embedding sub-network block. Each embedding block is a small vertical chain of layers shown inside a rounded dashed border:
- Green path: "Linear(100,256) → ReLU → Dropout(0.3) → Linear(256,512)"
- Orange path: "Linear(200,512) → ReLU → Dropout(0.3) → Linear(512,512)"
- Purple path: "Linear(50,512) → ReLU → Dropout(0.3) → Linear(512,512)"
- Red path: "Linear(17,128) → ReLU → Dropout(0.3) → Linear(128,512)"

Each embedding block outputs a box labeled "150×512维". Below each output box, draw a small "+" symbol with a circle icon labeled "模态嵌入" (Modality Embedding), indicating the addition of a learnable modality embedding vector.

**Stage 2 — "跨模态融合与位置编码层" (Cross-Modal Fusion & Positional Encoding Layer):**

The four "150×512维" outputs (each now with modality embedding) converge via four arrows into a single operation box labeled "均值池化融合 (Mean Pooling)", producing one box labeled "融合特征 (150×512)".

Below this box, draw a "+" symbol connecting to a box labeled "可学习位置编码 P ∈ R^{150×512}" (Learnable Positional Encoding). The result flows right into a single box labeled "编码输入 (150×512)".

**Stage 3 (center) — "Transformer时序编码层" (Transformer Temporal Encoding Layer):**

Draw a large vertically elongated rounded rectangle representing the Transformer Encoder stack. Inside this rectangle, show 6 identical stacked layers from bottom to top, each layer rendered as a smaller rectangle containing two sub-blocks:
- Sub-block A: "多头自注意力 (16头, d_k=32)" with a residual skip-connection arrow arching around it and a "LayerNorm" label.
- Sub-block B: "前馈网络 FFN (d_ff=2048)" with a residual skip-connection arrow arching around it and a "LayerNorm" label.
Label the full stack "6层 Transformer Encoder" along its left side. Mark "Dropout=0.1" as a small annotation.

From the top of the Transformer stack, draw an arrow into a box labeled "全局平均池化 (Global Average Pooling)", which outputs a single box labeled "融合向量 z (512维)".

**Stage 4 (rightmost) — "多任务预测层" (Multi-Task Prediction Layer):**

From the "融合向量 z (512维)" box, fan out five rightward arrows, each leading to an independent prediction head. The five heads are vertically stacked, each shown as a horizontal chain of small boxes in the same color:
- Head 1 (blue): "512 → 256 → ReLU → 256 → 128 → ReLU → 128 → 1 → Sigmoid → [1.0, 5.0]", final output label: "唇形同步 (lip_sync)"
- Head 2 (green): same structure, final output label: "表情自然度 (expression)"
- Head 3 (orange): same structure, final output label: "音频质量 (audio_quality)"
- Head 4 (purple): same structure, final output label: "跨模态一致性 (cross_modal)"
- Head 5 (red): same structure, final output label: "整体感知质量 (overall)"

Each head ends with a small output circle showing a score value like "3.8" as an example.

**Bottom annotation area — "训练优化机制" (Training Optimization Mechanism):**

At the very bottom of the diagram, spanning the full width, place a light gray background band containing two annotation boxes side by side:
- Left box (light yellow border): "动态任务权重策略" with three items listed: "① 固定权重", "② 不确定性自适应加权 (σ_k)", "③ GradNorm梯度均衡 (含5轮热身期)"
- Right box (light blue border): "标签掩码机制" with the note: "有效标签率: lip_sync 100% | expression 72.8% | audio_quality 77.8% | cross_modal 72.6% | overall 72.6%"
Between these two boxes, place a formula: "L_total = Σ(k=1 to 5) w_k · L_k"

Overall style: crisp vector-style lines, soft pastel fill colors for each modality (consistent color coding throughout), Chinese text labels in black, consistent font size, no 3D effects, suitable for insertion into an academic thesis. The diagram should be wide (landscape orientation) to accommodate the left-to-right four-stage layout.
