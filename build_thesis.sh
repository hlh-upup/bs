#!/bin/bash
# ============================================================
# 论文编译脚本 - 使用 Pandoc + pandoc-crossref
# ============================================================
# 
# 前提条件（安装方法见 README_BUILD.md）：
#   1. pandoc >= 3.0
#   2. pandoc-crossref
#   3. chinese-gb7714-2005-numeric.csl（引文样式文件）
#
# 用法：
#   chmod +x build_thesis.sh
#   ./build_thesis.sh              # 编译完整论文（docx）
#   ./build_thesis.sh pdf          # 编译为 PDF（需要 LaTeX）
#   ./build_thesis.sh chapters     # 仅编译当前有的章节
#
# ============================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查依赖
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}错误：未找到 $1，请先安装。${NC}"
        echo "安装方法请参见 README_BUILD.md"
        exit 1
    fi
}

echo -e "${GREEN}=== 论文编译脚本 ===${NC}"
echo ""

check_dependency pandoc

# 检查 pandoc-crossref
if ! pandoc-crossref --version &> /dev/null 2>&1; then
    echo -e "${YELLOW}警告：未找到 pandoc-crossref，交叉引用将不会生效${NC}"
    CROSSREF_FILTER=""
else
    CROSSREF_FILTER="--filter pandoc-crossref"
fi

# 下载 CSL 样式文件（如果不存在）
CSL_FILE="chinese-gb7714-2005-numeric.csl"
if [ ! -f "$CSL_FILE" ]; then
    echo -e "${YELLOW}下载 GB/T 7714 引文样式文件...${NC}"
    curl -sL "https://raw.githubusercontent.com/citation-style-language/styles/master/chinese-gb7714-2005-numeric.csl" -o "$CSL_FILE" 2>/dev/null || {
        echo -e "${YELLOW}无法下载 CSL 文件，将使用默认引文格式${NC}"
        CSL_OPT=""
    }
fi

if [ -f "$CSL_FILE" ]; then
    CSL_OPT="--csl=$CSL_FILE"
else
    CSL_OPT=""
fi

# 输出格式
OUTPUT_FORMAT="${1:-docx}"

# ============================================================
# 方式一：直接拼接各章节文件编译（推荐，最快捷）
# ============================================================

# 收集所有可用的章节文件（按顺序）
CHAPTER_FILES=""

if [ -f "chapter2.md" ]; then
    CHAPTER_FILES="$CHAPTER_FILES chapter2.md"
    echo "  ✓ 第1-2章: chapter2.md"
fi

if [ -f "chapter3.md" ]; then
    CHAPTER_FILES="$CHAPTER_FILES chapter3.md"
    echo "  ✓ 第3章: chapter3.md"
fi

if [ -f "chapter4.md" ]; then
    CHAPTER_FILES="$CHAPTER_FILES chapter4.md"
    echo "  ✓ 第4章: chapter4.md"
fi

if [ -f "chapter5.md" ]; then
    CHAPTER_FILES="$CHAPTER_FILES chapter5.md"
    echo "  ✓ 第5章: chapter5.md"
fi

if [ -f "chapter6.md" ]; then
    CHAPTER_FILES="$CHAPTER_FILES chapter6.md"
    echo "  ✓ 第6章: chapter6.md"
fi

if [ -z "$CHAPTER_FILES" ]; then
    echo -e "${RED}错误：未找到任何章节文件${NC}"
    exit 1
fi

echo ""

# 从 thesis_main.md 提取 YAML 元数据
METADATA_OPTS=""
if [ -f "thesis_main.md" ]; then
    METADATA_OPTS="--metadata-file=thesis_main.md"
fi

case "$OUTPUT_FORMAT" in
    docx)
        OUTPUT_FILE="thesis_output.docx"
        echo -e "${GREEN}编译 DOCX 格式...${NC}"
        
        # 如果有参考模板 docx，使用它
        REF_DOC_OPT=""
        if [ -f "reference.docx" ]; then
            REF_DOC_OPT="--reference-doc=reference.docx"
            echo "  使用参考模板: reference.docx"
        fi
        
        pandoc \
            $CHAPTER_FILES \
            -o "$OUTPUT_FILE" \
            --from markdown \
            --to docx \
            $CROSSREF_FILTER \
            --citeproc \
            --bibliography=references.bib \
            $CSL_OPT \
            $REF_DOC_OPT \
            --number-sections \
            --toc \
            --toc-depth=3 \
            -M lang=zh-CN \
            -M title="生成式数字人质量评价模型研究及系统构建" \
            -M reference-section-title="参考文献" \
            -M figureTitle="图" \
            -M tableTitle="表" \
            -M figPrefix="图" \
            -M tblPrefix="表" \
            -M secPrefix="" \
            -M chapters=true \
            2>&1
        ;;
    
    pdf)
        OUTPUT_FILE="thesis_output.pdf"
        echo -e "${GREEN}编译 PDF 格式（需要 XeLaTeX）...${NC}"
        
        check_dependency xelatex
        
        pandoc \
            $CHAPTER_FILES \
            -o "$OUTPUT_FILE" \
            --from markdown \
            --to pdf \
            --pdf-engine=xelatex \
            $CROSSREF_FILTER \
            --citeproc \
            --bibliography=references.bib \
            $CSL_OPT \
            --number-sections \
            --toc \
            --toc-depth=3 \
            -V CJKmainfont="SimSun" \
            -V mainfont="Times New Roman" \
            -V geometry:margin=2.5cm \
            -M lang=zh-CN \
            -M title="生成式数字人质量评价模型研究及系统构建" \
            -M reference-section-title="参考文献" \
            2>&1
        ;;
    
    *)
        echo -e "${RED}不支持的输出格式: $OUTPUT_FORMAT${NC}"
        echo "支持的格式: docx, pdf"
        exit 1
        ;;
esac

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo -e "${GREEN}✓ 编译成功！${NC}"
    echo -e "  输出文件: ${GREEN}$OUTPUT_FILE${NC}"
    echo -e "  文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
else
    echo -e "${RED}✗ 编译失败${NC}"
    exit 1
fi
