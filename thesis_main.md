---
title: "生成式数字人质量评价模型研究及系统构建"
title-en: "Research on Quality Evaluation Model of Generative Digital Humans and System Construction"
author: "韩立辉"
supervisor: ""
university: "中国传媒大学"
college: "数据科学与智能媒体学院"
major: "大数据技术与工程"
date: "2025"
lang: zh-CN
bibliography: references.bib
csl: chinese-gb7714-2005-numeric.csl
link-citations: true
reference-section-title: "参考文献"
figureTitle: "图"
tableTitle: "表"
listingTitle: "代码"
figPrefix: "图"
tblPrefix: "表"
lstPrefix: "代码"
secPrefix: "第"
eqnPrefix: "式"
chapters: true
chaptersDepth: 1
sectionsDepth: 3
autoSectionLabels: true
numberSections: true
---

<!-- 
  论文主文件 - 使用 pandoc + pandoc-crossref 编译
  
  编译方法请参见 build_thesis.sh 或 README_BUILD.md
  
  本文件按顺序包含所有章节。
  如果其他章节（第3-5章）也已转换为 Markdown，
  请在对应位置取消注释或添加内容。
-->

<!-- ==================== 第一章：绪论 ==================== -->
<!-- ==================== 第二章：相关技术基础与理论框架 ==================== -->
<!-- 第1-2章内容在 chapter2.md 中 -->

{{chapter2.md}}

<!-- ==================== 第三章：质量评价模型算法设计 ==================== -->
<!-- 
  TODO: 将第三章 Markdown 文件放在此处
  如果已有 chapter3.md，取消注释下面这行：
-->
<!-- {{chapter3.md}} -->

<!-- ==================== 第四章：实验验证与性能分析 ==================== -->
<!-- 
  TODO: 将第四章 Markdown 文件放在此处
  如果已有 chapter4.md，取消注释下面这行：
-->
<!-- {{chapter4.md}} -->

<!-- ==================== 第五章：集成质量评价闭环的智能授课视频生成系统 ==================== -->

{{chapter5.md}}

<!-- ==================== 第六章：总结与展望 ==================== -->

{{chapter6.md}}

<!-- ==================== 参考文献 ==================== -->
<!-- 参考文献由 pandoc-citeproc 从 references.bib 自动生成 -->

::: {#refs}
:::
