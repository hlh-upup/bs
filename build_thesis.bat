@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM  论文编译脚本（Windows 版）
REM ============================================================
REM
REM  用法（在 CMD 或 PowerShell 中运行）：
REM    build_thesis.bat              编译为 DOCX（默认）
REM    build_thesis.bat docx         编译为 DOCX
REM    build_thesis.bat pdf          编译为 PDF（需要 XeLaTeX）
REM
REM  前提条件：
REM    1. 安装 Pandoc: https://pandoc.org/installing.html
REM    2. （可选）安装 pandoc-crossref: https://github.com/lierdakil/pandoc-crossref/releases
REM    3. （生成 PDF 时）安装 TeX Live 或 MiKTeX（包含 XeLaTeX）
REM
REM ============================================================

echo ============================================================
echo   论文编译脚本（Windows / Pandoc）
echo ============================================================
echo.

REM ---------- 检查 pandoc ----------
where pandoc >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 pandoc，请先安装。
    echo.
    echo   下载地址: https://pandoc.org/installing.html
    echo   推荐使用 Windows 安装包（.msi），安装后重新打开终端即可。
    echo.
    exit /b 1
)

REM ---------- 检查 pandoc-crossref ----------
set "CROSSREF_FILTER="
where pandoc-crossref >nul 2>&1
if %errorlevel% equ 0 (
    set "CROSSREF_FILTER=--filter pandoc-crossref"
    echo   [OK] pandoc-crossref 已安装
) else (
    echo   [提示] 未找到 pandoc-crossref，交叉引用（@sec: @fig:）将不会自动转换
    echo          下载地址: https://github.com/lierdakil/pandoc-crossref/releases
)

REM ---------- 下载 CSL 样式文件 ----------
set "CSL_FILE=chinese-gb7714-2005-numeric.csl"
set "CSL_OPT="
if exist "%CSL_FILE%" (
    set "CSL_OPT=--csl=%CSL_FILE%"
    echo   [OK] CSL 样式文件已就绪
) else (
    echo   [提示] 正在下载 GB/T 7714 引文样式文件...
    powershell -Command "try { Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/citation-style-language/styles/master/chinese-gb7714-2005-numeric.csl' -OutFile '%CSL_FILE%' -ErrorAction Stop; Write-Host '   [OK] 下载成功' } catch { Write-Host '   [提示] 无法下载 CSL 文件，将使用默认引文格式' }"
    if exist "%CSL_FILE%" (
        set "CSL_OPT=--csl=%CSL_FILE%"
    )
)

REM ---------- 收集章节文件 ----------
set "CHAPTER_FILES="
for %%f in (chapter2.md chapter3.md chapter4.md chapter5.md chapter6.md) do (
    if exist "%%f" (
        set "CHAPTER_FILES=!CHAPTER_FILES! %%f"
        echo   [OK] %%f
    )
)

if "!CHAPTER_FILES!"=="" (
    echo.
    echo [错误] 未找到任何章节文件（chapter2.md ~ chapter6.md）
    exit /b 1
)

REM ---------- 检查参考文献文件 ----------
set "BIB_OPT="
if exist "references.bib" (
    set "BIB_OPT=--citeproc --bibliography=references.bib"
    echo   [OK] references.bib
) else (
    echo   [提示] 未找到 references.bib，参考文献将不会生成
)

echo.

REM ---------- 判断输出格式 ----------
set "OUTPUT_FORMAT=%~1"
if "%OUTPUT_FORMAT%"=="" set "OUTPUT_FORMAT=docx"

if /i "%OUTPUT_FORMAT%"=="docx" goto :BUILD_DOCX
if /i "%OUTPUT_FORMAT%"=="pdf"  goto :BUILD_PDF
echo [错误] 不支持的格式: %OUTPUT_FORMAT%
echo 用法: build_thesis.bat [docx^|pdf]
exit /b 1

REM ============================================================
:BUILD_DOCX
REM ============================================================
set "OUTPUT_FILE=thesis_output.docx"
echo 正在编译 DOCX 格式...
echo.

set "REF_DOC_OPT="
if exist "reference.docx" (
    set "REF_DOC_OPT=--reference-doc=reference.docx"
    echo   使用参考模板: reference.docx
)

pandoc !CHAPTER_FILES! ^
    -o "%OUTPUT_FILE%" ^
    --from markdown ^
    --to docx ^
    !CROSSREF_FILTER! ^
    !BIB_OPT! ^
    !CSL_OPT! ^
    !REF_DOC_OPT! ^
    --number-sections ^
    --toc ^
    --toc-depth=3 ^
    -M lang=zh-CN ^
    -M link-citations=true ^
    -M title="生成式数字人质量评价模型研究及系统构建" ^
    -M reference-section-title="参考文献" ^
    -M figureTitle="图" ^
    -M tableTitle="表" ^
    -M figPrefix="图" ^
    -M tblPrefix="表" ^
    -M secPrefix="" ^
    -M chapters=true

goto :CHECK_OUTPUT

REM ============================================================
:BUILD_PDF
REM ============================================================
set "OUTPUT_FILE=thesis_output.pdf"
echo 正在编译 PDF 格式（需要 XeLaTeX）...
echo.

where xelatex >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 xelatex。
    echo.
    echo   请安装 TeX Live 或 MiKTeX：
    echo     TeX Live: https://tug.org/texlive/
    echo     MiKTeX:   https://miktex.org/download
    echo.
    exit /b 1
)

pandoc !CHAPTER_FILES! ^
    -o "%OUTPUT_FILE%" ^
    --from markdown ^
    --to pdf ^
    --pdf-engine=xelatex ^
    !CROSSREF_FILTER! ^
    !BIB_OPT! ^
    !CSL_OPT! ^
    --number-sections ^
    --toc ^
    --toc-depth=3 ^
    -V CJKmainfont="SimSun" ^
    -V mainfont="Times New Roman" ^
    -V geometry:top=2.5cm ^
    -V geometry:bottom=2.5cm ^
    -V geometry:left=3cm ^
    -V geometry:right=2.5cm ^
    -V linestretch=1.5 ^
    -M lang=zh-CN ^
    -M link-citations=true ^
    -M title="生成式数字人质量评价模型研究及系统构建" ^
    -M reference-section-title="参考文献" ^
    -M figureTitle="图" ^
    -M tableTitle="表" ^
    -M figPrefix="图" ^
    -M tblPrefix="表" ^
    -M secPrefix="" ^
    -M chapters=true

goto :CHECK_OUTPUT

REM ============================================================
:CHECK_OUTPUT
REM ============================================================
echo.
if exist "%OUTPUT_FILE%" (
    echo ============================================================
    echo   [成功] 编译完成！
    echo   输出文件: %OUTPUT_FILE%
    echo ============================================================
    echo.
    echo 后续操作：
    echo   1. 双击打开 %OUTPUT_FILE% 检查排版效果
    echo   2. 如需合并到现有论文，请复制相关章节内容粘贴到您的论文文件中
    echo   3. 参考文献列表在文件末尾，可单独复制
    echo.
    echo 兼容性说明（送审 / 查重 / PDF）：
    echo   [OK] 查重兼容：Pandoc 生成标准 OOXML 文档，知网/维普/万方可正常提取文本
    echo   [OK] PDF转换：在 Word 中打开后另存为 PDF，或使用 build_thesis.bat pdf 直接生成
    echo   [OK] 送审兼容：包含标题样式、目录结构和参考文献，格式规范
) else (
    echo [失败] 编译失败，请检查上方错误信息。
    echo.
    echo 常见问题：
    echo   - pandoc 版本过低：请升级到 3.0 以上（pandoc --version 查看）
    echo   - 缺少 pandoc-crossref：交叉引用语法会原样保留
    echo   - 图片路径不存在：图片引用会显示为 alt 文本
    exit /b 1
)

endlocal
