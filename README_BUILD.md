# 论文交叉引用与合并编译指南

## 最快捷的方法

### 方法一：使用 Pandoc 编译（推荐，全自动）

**只需三步**即可将所有 Markdown 章节合并为一个完整的 Word 论文文件：

```bash
# 1. 安装 pandoc 和 pandoc-crossref
#    macOS:
brew install pandoc pandoc-crossref
#    Ubuntu/Debian:
sudo apt install pandoc && pip install pandoc-crossref
#    Windows (使用 scoop):
scoop install pandoc pandoc-crossref

# 2. 运行编译脚本
chmod +x build_thesis.sh
./build_thesis.sh

# 3. 输出文件：thesis_output.docx
```

编译脚本会自动：
- 按顺序合并所有章节（第1-6章）
- 处理交叉引用（`@sec:...`、`@fig:...`）
- 生成 GB/T 7714 格式的参考文献列表
- 生成目录
- 自动编号章节和图表

### 方法二：手动复制到 Word（最简单）

如果不想安装 Pandoc，可以直接将 Markdown 内容复制到现有的 `.docx` 模板中：

1. 打开 `学位论文-韩立辉(2).docx` 模板
2. 按顺序复制各 `.md` 文件内容到对应章节位置
3. 在 Word 中手动添加交叉引用（插入 → 交叉引用）

---

## 文件结构说明

```
├── chapter2.md          # 第1章（绪论）+ 第2章（相关技术基础）
├── chapter3.md          # 第3章（待添加）
├── chapter4.md          # 第4章（待添加）
├── chapter5.md          # 第5章（待添加）
├── chapter6.md          # 第6章（总结与展望）
├── references.bib       # BibTeX 参考文献库（58条）
├── thesis_main.md       # 论文合并主文件（YAML 元数据 + 章节包含）
├── build_thesis.sh      # 一键编译脚本
└── README_BUILD.md      # 本说明文件
```

## 交叉引用语法说明

本项目使用 [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref) 的交叉引用语法。

### 章节引用

每个章节标题后都添加了 `{#sec:label}` 锚点标签：

```markdown
# 1 绪论 {#sec:ch1}
## 1.1 研究背景及意义 {#sec:background}
# 2 相关技术基础与理论框架 {#sec:ch2}
## 2.1 说话人脸视频生成技术 {#sec:gen-tech}
```

在正文中引用章节：

```markdown
详见 @sec:ch2                    → "详见 第2章"
如 @sec:gen-tech 所述              → "如 第2.1节 所述"
技术细节详见 @sec:ch2 与第三章     → "技术细节详见 第2章 与第三章"
```

### 图片引用

每个图片都添加了 `{#fig:label}` 标签：

```markdown
![图2.1 Wav2Lip网络架构图](media/wav2lip_architecture.png){#fig:wav2lip}
```

在正文中引用图片：

```markdown
如 @fig:wav2lip 所示    → "如 图2.1 所示"
```

### 参考文献引用

所有 `[N]` 编号引用已转换为 BibTeX 引用键：

```markdown
[@Goodfellow2014]                          → "[1]"（对应 GAN 原论文）
[@Prajwal2020]                             → "[2]"（对应 Wav2Lip）
[@Toshpulatov2023; @Bai2025]              → "[8,9]"（多引用合并）
```

### 当前已定义的标签一览

| 标签 | 对应内容 |
|------|---------|
| `@sec:ch1` | 第1章 绪论 |
| `@sec:background` | 1.1 研究背景及意义 |
| `@sec:related-work` | 1.2 国内外研究现状 |
| `@sec:intl-research` | 1.2.1 国际研究现状 |
| `@sec:cn-research` | 1.2.2 国内研究现状 |
| `@sec:structure` | 1.3 研究内容与结构安排 |
| `@sec:ch2` | 第2章 相关技术基础与理论框架 |
| `@sec:gen-tech` | 2.1 说话人脸视频生成技术 |
| `@sec:gan-wav2lip` | 2.1.1 基于GAN的方法与Wav2Lip |
| `@sec:3d-sadtalker` | 2.1.2 基于3D感知的方法与SadTalker |
| `@sec:diffusion-transformer` | 2.1.3 扩散模型、Transformer与少样本 |
| `@sec:qa-theory` | 2.2 质量评估与多模态分析理论 |
| `@sec:qa-system` | 2.2.1 数字人视频质量评估体系 |
| `@sec:multimodal-fusion` | 2.2.2 多模态特征融合技术 |
| `@sec:mtl` | 2.2.3 多任务学习优化策略 |
| `@sec:tts-arch` | 2.3 语音合成与系统架构技术 |
| `@sec:voice-clone` | 2.3.1 少样本语音克隆与情感控制 |
| `@sec:bs-arch` | 2.3.2 前后端分离架构与工程优化 |
| `@sec:ch2-summary` | 2.4 本章小结 |
| `@sec:ch6` | 第6章 总结与展望 |
| `@sec:conclusion` | 6.1 论文工作总结 |
| `@sec:innovation` | 6.2 主要创新点 |
| `@sec:future` | 6.3 未来研究展望 |
| `@fig:wav2lip` | 图2.1 Wav2Lip网络架构图 |
| `@fig:sadtalker` | 图2.2 SadTalker系统架构图 |
| `@fig:voice-clone` | 图2.3 少样本语音克隆技术流程图 |
| `@fig:qa-framework` | 图2.4 多维度质量评估体系框架图 |
| `@fig:fusion-compare` | 图2.5 多模态特征融合架构对比图 |
| `@fig:mtl-weight` | 图2.6 多任务学习权重动态调整示意图 |
| `@fig:tts-evolution` | 图2.7 语音合成技术演进路线图 |
| `@fig:system-arch` | 图2.8 前后端分离系统架构图 |

## 添加新章节的步骤

当您准备好第3-5章的内容时：

1. **创建 Markdown 文件**（如 `chapter3.md`）
2. **添加章节标签**：

```markdown
# 3 质量评价模型算法设计 {#sec:ch3}

## 3.1 模型总体架构 {#sec:model-arch}

如 @sec:gen-tech 所述的说话人脸视频生成技术...

![图3.1 模型架构图](media/model_architecture.png){#fig:model-arch}

如 @fig:model-arch 所示，模型包含...
```

3. **添加参考文献**（如有新引用）到 `references.bib`
4. **重新编译**：`./build_thesis.sh`

## 参考文献格式

`references.bib` 中的每条记录使用标准 BibTeX 格式，同时在 `note` 字段保存了完整的 GB/T 7714 格式引文：

```bibtex
@inproceedings{Goodfellow2014,
  author = {Goodfellow I J, Pouget-Abadie J, Mirza M, et al},
  title = {Generative Adversarial Networks},
  year = {2014},
  note = {Goodfellow I J, ... [C]//Advances in NeurIPS. 2014: 2672-2680.}
}
```

编译时使用 `chinese-gb7714-2005-numeric.csl` 样式文件确保输出符合国标格式。

## 常见问题

**Q: 编译后交叉引用显示为 "??" 怎么办？**  
A: 确保安装了 `pandoc-crossref`，并且标签名称完全匹配（区分大小写）。

**Q: 如何生成 PDF？**  
A: 运行 `./build_thesis.sh pdf`，需要先安装 XeLaTeX（如 TeX Live）。

**Q: 如何自定义 Word 输出样式？**  
A: 创建一个 `reference.docx` 模板文件（在 Word 中设置好字体、行距、标题样式），编译时会自动使用。

**Q: 图片路径找不到？**  
A: 确保 `media/` 文件夹中包含所有引用的图片文件，或调整图片路径。
