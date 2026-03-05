# 论文交叉引用与合并编译指南

## 最快捷的方法（推荐）

### 方法一：Python 直接生成 Word（最推荐，无需安装额外工具）

**只需两步**即可生成符合中国计算机类硕士论文格式的 Word 文档：

```bash
# 1. 安装依赖（仅首次）
pip install python-docx

# 2. 生成 Word 文档
python3 generate_docx.py
```

**输出**：`thesis_chapters.docx`

生成的 Word 文件特点：
- ✅ **参考文献**：符合 **GB/T 7714-2015** 国家标准格式
- ✅ **引用编号**：正文中的 `[@key]` 自动转为 `[1]`、`[2]` 等编号
- ✅ **交叉引用**：`@sec:ch2` 自动转为 "第二章"，`@fig:wav2lip` 转为 "图2.1"
- ✅ **字体格式**：
  - 章标题：黑体 三号（16pt）居中
  - 节标题：黑体 四号（14pt）
  - 小节标题：黑体 小四号（12pt）
  - 正文：宋体 小四号（12pt），英文 Times New Roman
  - 参考文献：宋体 五号（10.5pt）
- ✅ **页面设置**：上下 2.5cm、左 3cm、右 2.5cm，1.5 倍行距
- ✅ **首行缩进**：正文段落自动首行缩进 2 字符

**合并到现有论文**：

1. 打开生成的 `thesis_chapters.docx`
2. 选择需要的章节内容，复制
3. 粘贴到您已有的 `学位论文-韩立辉(2).docx` 对应位置
4. 参考文献列表在文件末尾，可单独复制到论文末尾

### 方法二：使用 Pandoc 编译

#### Windows 用户（推荐）

**第一步：安装 Pandoc**

1. 访问 https://pandoc.org/installing.html
2. 下载 Windows 安装包（`.msi`），双击安装
3. 安装完成后**重新打开**终端（CMD 或 PowerShell），验证安装：

```cmd
pandoc --version
```

**第二步（可选）：安装 pandoc-crossref**（支持 `@sec:` `@fig:` 交叉引用）

1. 访问 https://github.com/lierdakil/pandoc-crossref/releases
2. 下载与 Pandoc 版本匹配的 Windows 版本（`pandoc-crossref-Windows.7z`）
3. 解压 `pandoc-crossref.exe` 到 Pandoc 安装目录（通常 `C:\Users\你的用户名\AppData\Local\Pandoc\`）

**第三步：编译**

```cmd
REM 在论文项目根目录下打开 CMD，运行：

REM 生成 Word 文档
build_thesis.bat

REM 生成 PDF（需要额外安装 TeX Live 或 MiKTeX）
build_thesis.bat pdf
```

**输出**：`thesis_output.docx` 或 `thesis_output.pdf`

> 💡 **提示**：如果不想安装 pandoc-crossref，脚本仍可正常运行，只是 `@sec:ch2`、`@fig:wav2lip` 等交叉引用会原样保留，需要手动替换为"第二章"、"图2.1"等。

#### macOS / Linux 用户

```bash
# 安装 pandoc
brew install pandoc pandoc-crossref    # macOS
sudo apt install pandoc                # Ubuntu

# 编译
./build_thesis.sh pandoc               # 生成 DOCX
./build_thesis.sh pandoc pdf           # 生成 PDF
```

### 方法三：手动复制到 Word

直接将 Markdown 内容复制到现有的 `.docx` 模板中，手动调整格式。

---

## 文件结构说明

```
├── chapter2.md          # 第1章（绪论）+ 第2章（相关技术基础）
├── chapter3.md          # 第3章（待添加）
├── chapter4.md          # 第4章（待添加）
├── chapter5.md          # 第5章（待添加）
├── chapter6.md          # 第6章（总结与展望）
├── references.bib       # BibTeX 参考文献库（58条）
├── generate_docx.py     # ★ Python Word 生成脚本（推荐使用）
├── thesis_main.md       # Pandoc 合并主文件（YAML 元数据）
├── build_thesis.sh      # 编译脚本 - macOS / Linux
├── build_thesis.bat     # 编译脚本 - Windows（Pandoc 方式）
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

**Q: 最快的方式是什么？**
A: 运行 `pip install python-docx && python3 generate_docx.py`，两条命令即可生成格式正确的 Word 文件。

**Q: Windows 上如何用 Pandoc 编译？**
A: 安装 Pandoc 后，在项目目录打开 CMD，运行 `build_thesis.bat`（生成 DOCX）或 `build_thesis.bat pdf`（生成 PDF）。

**Q: 如何合并到我已有的论文 Word 文件？**
A: 打开生成的 `thesis_chapters.docx` 或 `thesis_output.docx`，选择需要的章节内容复制，粘贴到您的论文中。参考文献在文末，可单独复制。

**Q: 参考文献格式是否符合 GB/T 7714？**
A: 是的。`generate_docx.py` 中的参考文献已按 GB/T 7714-2015 国家标准格式存储，包括期刊[J]、会议[C]、学位论文[D]、标准[S]、电子资源[EB/OL]等所有类型。Pandoc 方式使用 `chinese-gb7714-2005-numeric.csl` 样式文件。

**Q: 编译后交叉引用显示为 "??" 怎么办？**
A: 使用 `generate_docx.py`（Python 方式）不会有此问题。如果使用 Pandoc 方式，请确保安装了 `pandoc-crossref`。

**Q: 如何生成 PDF？**
A: Windows: `build_thesis.bat pdf`；macOS/Linux: `./build_thesis.sh pandoc pdf`。都需要先安装 XeLaTeX（TeX Live 或 MiKTeX）。

**Q: 送审和查重会不会有问题？**
A: 不会。无论 Python 方式还是 Pandoc 方式生成的 `.docx` 都是标准 OOXML 格式，知网/维普/万方等查重系统可正常提取文本内容。转 PDF 后也不会出现乱码或格式丢失问题。

**Q: 如何自定义 Word 输出样式？**
A: 使用 Pandoc 方式时，创建一个 `reference.docx` 模板文件（参考 [Pandoc 文档](https://pandoc.org/MANUAL.html#option--reference-doc)）。使用 Python 方式时，可直接修改 `generate_docx.py` 中的字体和字号设置。

**Q: 图片路径找不到？**
A: 确保 `media/` 文件夹中包含所有引用的图片文件，或调整图片路径。Python 方式会自动显示占位符。
