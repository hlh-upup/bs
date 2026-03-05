#!/bin/bash
# ============================================================
# 论文编译脚本
# ============================================================
#
# 用法：
#   chmod +x build_thesis.sh
#   ./build_thesis.sh              # 推荐：使用 Python 生成格式化 docx
#   ./build_thesis.sh python       # 同上
#   ./build_thesis.sh pandoc       # 使用 Pandoc 编译（需要额外安装）
#   ./build_thesis.sh pandoc pdf   # 使用 Pandoc 编译 PDF（需要 LaTeX）
#
# 方式一（推荐）：Python + python-docx
#   前提：pip install python-docx
#   特点：无需安装 pandoc，格式完全符合中国硕士论文规范
#
# 方式二：Pandoc + pandoc-crossref
#   前提：pandoc >= 3.0, pandoc-crossref
#
# ============================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 论文编译脚本 ===${NC}"
echo ""

# 决定编译方式
BUILD_MODE="${1:-python}"

case "$BUILD_MODE" in
    python)
        # ==============================================================
        # 方式一：Python 生成（推荐，格式最准确）
        # ==============================================================
        echo -e "${GREEN}使用 Python 方式生成 Word 文档...${NC}"
        echo "  格式：中国计算机类硕士论文 / GB/T 7714-2015 参考文献"
        echo ""

        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}错误：未找到 python3${NC}"
            exit 1
        fi

        # 检查 python-docx
        if ! python3 -c "import docx" 2>/dev/null; then
            echo -e "${YELLOW}安装 python-docx...${NC}"
            pip3 install python-docx
        fi

        python3 generate_docx.py
        ;;

    pandoc|pandoc-docx)
        # ==============================================================
        # 方式二：Pandoc 编译
        # ==============================================================
        if ! command -v pandoc &> /dev/null; then
            echo -e "${RED}错误：未找到 pandoc，请先安装。${NC}"
            echo "  macOS: brew install pandoc"
            echo "  Ubuntu: sudo apt install pandoc"
            exit 1
        fi

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

        OUTPUT_FORMAT="${2:-docx}"

        # 收集可用章节文件
        CHAPTER_FILES=""
        for ch in chapter2.md chapter3.md chapter4.md chapter5.md chapter6.md; do
            if [ -f "$ch" ]; then
                CHAPTER_FILES="$CHAPTER_FILES $ch"
                echo "  ✓ $ch"
            fi
        done

        if [ -z "$CHAPTER_FILES" ]; then
            echo -e "${RED}错误：未找到任何章节文件${NC}"
            exit 1
        fi
        echo ""

        case "$OUTPUT_FORMAT" in
            docx)
                OUTPUT_FILE="thesis_output.docx"
                echo -e "${GREEN}编译 DOCX 格式...${NC}"

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
                    -M link-citations=true \
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

                if ! command -v xelatex &> /dev/null; then
                    echo -e "${RED}错误：未找到 xelatex${NC}"
                    exit 1
                fi

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
                    -M link-citations=true \
                    -M title="生成式数字人质量评价模型研究及系统构建" \
                    -M reference-section-title="参考文献" \
                    2>&1
                ;;

            *)
                echo -e "${RED}不支持的格式: $OUTPUT_FORMAT${NC}"
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
        ;;

    *)
        echo "用法: $0 [python|pandoc] [docx|pdf]"
        echo ""
        echo "  python   - 推荐，使用 Python 生成格式化 Word（默认）"
        echo "  pandoc   - 使用 Pandoc 编译"
        exit 1
        ;;
esac
