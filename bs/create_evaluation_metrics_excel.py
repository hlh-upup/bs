#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成主观评价实验指标Excel表格
包含详细的指标介绍、评分标准和实验设计说明
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_evaluation_metrics_excel():
    """创建评价指标Excel表格"""
    
    # 1. 基本指标信息表
    basic_metrics = {
        '指标ID': ['MET001', 'MET002', 'MET003', 'MET004', 'MET005'],
        '指标名称': ['唇音同步质量', '表情自然度', '音频质量', '视觉清晰度', '整体质量'],
        '指标类型': ['技术指标', '感知指标', '技术指标', '感知指标', '综合指标'],
        '评价维度': ['时序同步', '自然程度', '信号质量', '画面质量', '总体印象'],
        '权重建议': ['30%', '25%', '20%', '15%', '10%'],
        '数据类型': ['5分李克特量表', '5分李克特量表', '5分李克特量表', '5分李克特量表', '5分李克特量表'],
        '必填项': ['是', '是', '是', '是', '是']
    }
    
    # 2. 详细指标说明表
    detailed_descriptions = {
        '指标ID': ['MET001', 'MET002', 'MET003', 'MET004', 'MET005'],
        '科学定义': [
            '视频中嘴唇动作与对应语音信号在时间上的吻合程度，衡量AI生成的视觉-音频时序对齐精度',
            '面部表情变化的流畅性、连贯性和真实性，反映AI模拟人类表情的自然程度',
            '语音信号的清晰度、可懂度和保真度，评估AI生成或处理音频的技术质量',
            '视频画面的分辨率、清晰度和细节表现，评估AI生成视频的视觉质量',
            '对视频整体质量的主观综合评价，包含所有维度的总体印象'
        ],
        '评价重点': [
            ' lipsync精度、开口时机、语音时长匹配',
            '表情连贯性、肌肉运动自然度、情感表达真实性',
            '语音清晰度、噪音水平、音频失真程度',
            '画面锐利度、边缘清晰度、细节保留程度',
            '整体观感、舒适度、真实感'
        ],
        '应用场景': [
            '视频会议、虚拟主播、配音视频',
            '情感计算、人机交互、娱乐应用',
            '语音合成、音频处理、通讯应用',
            '视频生成、图像处理、显示技术',
            '产品评估、用户体验研究'
        ],
        '技术关联': [
            '时序对齐算法、音视频同步技术',
            '面部动画、表情合成、情感计算',
            '语音编码、降噪算法、音频增强',
            '超分辨率、图像生成、压缩算法',
            '多模态融合、质量评估模型'
        ]
    }
    
    # 3. 评分标准表
    scoring_criteria = {
        '指标': ['唇音同步质量', '表情自然度', '音频质量', '视觉清晰度', '整体质量'],
        '1分-很差': [
            '嘴唇动作与语音完全不同步，存在明显延迟或超前',
            '表情僵硬、机械，动作不连贯，明显虚假',
            '语音模糊不清，噪音严重，难以辨认内容',
            '画面模糊，细节丢失，边缘不清晰',
            '所有方面都很差，无法接受的质量'
        ],
        '2分-较差': [
            '嘴唇动作与语音有较大不同步，容易察觉',
            '表情不够自然，动作有些生硬',
            '语音可懂但不够清晰，有一定噪音',
            '画面较模糊，细节不够清晰',
            '多数方面较差，整体质量低'
        ],
        '3分-一般': [
            '嘴唇动作与语音基本同步，偶有小问题',
            '表情基本自然，偶有不自然之处',
            '语音清晰度一般，有一定质量',
            '画面清晰度一般，基本可接受',
            '质量中等，可以接受的水平'
        ],
        '4分-良好': [
            '嘴唇动作与语音同步良好，很难察觉问题',
            '表情自然流畅，动作合理',
            '语音清晰，噪音很少',
            '画面清晰，细节表现良好',
            '质量良好，令人满意'
        ],
        '5分-优秀': [
            '嘴唇动作与语音完美同步，如同真人',
            '表情非常自然，难以区分与真人的差异',
            '语音非常清晰，无任何噪音干扰',
            '画面非常清晰，细节丰富',
            '所有方面都很优秀，完美质量'
        ]
    }
    
    # 4. 实验设计信息表
    experiment_design = {
        '设计要素': [
            '实验类型',
            '设计方法',
            '样本数量',
            '视频对数量',
            '评价方式',
            '盲化设计',
            '平衡设计',
            '质量控制',
            '预计时长',
            '目标人群'
        ],
        '具体内容': [
            '被试内设计 (Within-subjects design)',
            '配对比较法 (Paired comparison)',
            '30名参与者',
            '20个视频对',
            '在线Web界面评价',
            'A/B标签盲化',
            '呈现顺序随机化',
            '注意力检查题、时间监控',
            '20-30分钟/人',
            '18-45岁，不同背景'
        ],
        '科学依据': [
            '减少个体差异影响，提高统计效力',
            '直接对比两种模型效果，差异更明显',
            '满足统计学基本样本量要求',
            '覆盖不同质量层次，具有代表性',
            '便于数据收集和自动化处理',
            '避免主观偏见，提高数据客观性',
            '消除顺序效应和疲劳效应',
            '确保数据质量和参与者的专注度',
            '符合注意力持续时间，避免疲劳',
            '覆盖不同用户群体，结果具有普适性'
        ]
    }
    
    # 5. 数据收集计划表
    data_collection = {
        '数据类型': [
            '基本信息数据',
            '评价评分数据',
            '偏好比较数据',
            '时间戳数据',
            '质量监控数据',
            '开放式反馈'
        ],
        '收集内容': [
            '年龄、性别、教育程度、技术经验',
            '5个维度的1-5分评分',
            'A/B偏好选择和理由',
            '开始时间、结束时间、评价时长',
            '完成率、注意力检查结果',
            '文字评论、改进建议'
        ],
        '数据格式': [
            '分类变量、有序变量',
            '数值型 (1-5)',
            '分类变量 + 文本',
            '时间戳、持续时间',
            '布尔值、百分比',
            '字符串、文本'
        ],
        '分析用途': [
            '人群分析、分组比较',
            '描述统计、假设检验',
            '偏好分析、相关性分析',
            '行为分析、质量控制',
            '数据筛选、权重调整',
            '质性分析、深度理解'
        ]
    }
    
    # 6. 统计分析计划表
    analysis_plan = {
        '分析方法': [
            '描述性统计',
            '配对t检验',
            'Wilcoxon符号秩检验',
            '效应量计算',
            '一致性分析',
            '相关性分析',
            '多变量分析',
            '可视化分析'
        ],
        '应用场景': [
            '均值、标准差、分布特征',
            '原始模型vs优化模型差异检验',
            '非参数配对比较',
            'Cohen\'s d、Cliff\'s delta',
            'Cronbach\'s α、ICC',
            '维度间相关关系',
            'PCA、因子分析',
            '箱线图、热力图、雷达图'
        ],
        '预期结果': [
            '基本统计特征概览',
            '差异显著性检验结果',
            '稳健性验证结果',
            '改进程度的量化指标',
            '测量工具信度指标',
            '指标间关联模式',
            '潜在结构识别',
            '直观的结果展示'
        ],
        '软件工具': [
            'Python (pandas, numpy)',
            'scipy.stats',
            'scipy.stats',
            'scipy.stats, effectsize',
            'pingouin, sklearn',
            'pandas, scipy',
            'sklearn, factor_analyzer',
            'matplotlib, seaborn, plotly'
        ]
    }
    
    # 7. 质量控制措施表
    quality_control = {
        '控制点': [
            '数据收集前',
            '数据收集时',
            '数据收集后',
            '数据分析前',
            '数据分析中',
            '结果报告时'
        ],
        '控制措施': [
            '预实验、仪器校准、界面测试',
            '标准化指导语、练习试验、时间监控',
            '数据清洗、异常值检测、完整性检查',
            '正态性检验、方差齐性检验、样本量检验',
            '多重比较校正、离群值处理、敏感性分析',
            '效应量报告、置信区间、可视化验证'
        ],
        '质量标准': [
            '实验材料无误、界面运行正常',
            '指导语一致、参与者理解任务',
            '数据完整、格式正确、逻辑一致',
            '满足统计假设、样本量充足',
            '结果稳健、方法适当',
            '报告全面、解释合理'
        ],
        '负责人': [
            '实验设计员',
            '实验管理员',
            '数据质量员',
            '统计分析师',
            '统计分析师',
            '研究负责人'
        ]
    }
    
    # 创建Excel文件
    output_path = Path('subjective_experiment/主观评价实验指标详解.xlsx')
    output_path.parent.mkdir(exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 写入各个表格
        pd.DataFrame(basic_metrics).to_excel(writer, sheet_name='基本指标信息', index=False)
        pd.DataFrame(detailed_descriptions).to_excel(writer, sheet_name='详细指标说明', index=False)
        pd.DataFrame(scoring_criteria).to_excel(writer, sheet_name='评分标准', index=False)
        pd.DataFrame(experiment_design).to_excel(writer, sheet_name='实验设计', index=False)
        pd.DataFrame(data_collection).to_excel(writer, sheet_name='数据收集计划', index=False)
        pd.DataFrame(analysis_plan).to_excel(writer, sheet_name='统计分析计划', index=False)
        pd.DataFrame(quality_control).to_excel(writer, sheet_name='质量控制措施', index=False)
        
        # 调整列宽
        for sheet_name in writer.book.sheetnames:
            worksheet = writer.book[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    return output_path

def create_experiment_summary():
    """创建实验总结报告"""
    
    summary = {
        'title': 'AI生成说话人脸视频主观评价实验',
        'version': '1.0',
        'date': '2025-01-17',
        'overview': {
            'purpose': '通过主观评价方法对比原始模型和优化模型生成的说话人脸视频质量差异',
            'scope': '涵盖5个关键维度的主观评价，采用科学的实验设计方法',
            'duration': '预计2-3周完成数据收集，1周完成数据分析'
        },
        'key_features': [
            '科学的被试内实验设计',
            '5维度综合评价体系',
            '智能视频选取算法',
            '在线数据收集平台',
            '完整的质量控制流程',
            '详细的统计分析计划'
        ],
        'expected_outcomes': [
            '量化原始模型vs优化模型的用户感知差异',
            '识别各维度的改进程度和优先级',
            '建立主观评价与客观指标的相关性',
            '为后续优化提供科学依据'
        ],
        'risks_mitigation': [
            {
                'risk': '参与者招募困难',
                'mitigation': '多渠道招募，适当激励措施'
            },
            {
                'risk': '数据质量不佳',
                'mitigation': '严格的质量控制，注意力检查'
            },
            {
                'risk': '样本代表性不足',
                'mitigation': '多样化招募策略，分层抽样'
            },
            {
                'risk': '技术故障',
                'mitigation': '充分测试，备份方案'
            }
        ]
    }
    
    # 保存总结
    summary_path = Path('subjective_experiment/experiment_summary.json')
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary_path

def main():
    """主函数"""
    print("生成主观评价实验指标Excel表格...")
    
    # 创建Excel表格
    excel_path = create_evaluation_metrics_excel()
    print(f"Excel表格已生成: {excel_path}")
    
    # 创建实验总结
    summary_path = create_experiment_summary()
    print(f"实验总结已生成: {summary_path}")
    
    print("\n表格内容说明:")
    print("1. 基本指标信息 - 指标概览和基本信息")
    print("2. 详细指标说明 - 科学定义和技术关联")
    print("3. 评分标准 - 1-5分的详细评分说明")
    print("4. 实验设计 - 科学的实验设计方法")
    print("5. 数据收集计划 - 系统的数据收集方案")
    print("6. 统计分析计划 - 完整的统计分析策略")
    print("7. 质量控制措施 - 全流程的质量保证")
    
    print(f"\n文件位置: {excel_path.parent}")
    print("使用建议:")
    print("- 仔细阅读评分标准，确保理解每个维度的含义")
    print("- 根据实验设计表准备实验材料")
    print("- 参考统计分析计划进行数据处理")
    print("- 遵循质量控制措施确保数据质量")

if __name__ == "__main__":
    main()