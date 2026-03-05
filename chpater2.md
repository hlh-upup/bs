# 1 绪论

## 1.1 研究背景及意义

随着人工智能（Artificial Intelligence, AI）、计算机视觉（Computer Vision, CV）和生成式模型技术的迅猛发展，生成式数字人技术已成为数字媒体、人机交互、虚拟现实（VR）、增强现实（AR）以及元宇宙领域的前沿热点。该技术通过音频信号驱动静态人像照片或视频帧，自动合成具有自然唇形同步、面部表情变化、头部姿态运动以及语音情感表达的说话人脸视频（Talking Face Video）。这种视频生成能力为虚拟主播、在线教育、虚拟助手、电影后期配音、游戏角色互动、政务服务机器人以及医疗康复训练等众多场景提供了高效、智能且低成本的视觉呈现手段。生成式数字人不仅能够大幅降低传统视频制作的人力、时间和经济成本，还能实现高度个性化的实时交互体验，已成为数字经济与数字社会建设的重要技术支撑。

在全球范围内，数字人技术正从实验室概念验证阶段加速迈向大规模商业化应用。根据国际权威市场研究机构Statista和Gartner的最新报告显示，2023年至2025年间，全球数字人市场规模以年复合增长率超过42%的速度高速扩张，预计到2026年将突破1200亿美元大关。这一爆炸式增长得益于深度学习技术的突破性进展，生成对抗网络（GAN）[21]、神经辐射场（NeRF）、扩散模型（Diffusion Model）以及多模态大语言模型等关键技术（详见第二章）推动了生成视频的真实感、泛化能力和渲染效率的质的飞跃。然而，尽管底层生成技术日臻成熟，现有生成式数字人视频在实际部署过程中仍面临多维度感知质量瓶颈：唇形与语音时序同步精度不足、面部表情自然度欠佳、音频与视觉跨模态一致性较差、整体感知质量难以满足高标准应用场景。这些问题不仅直接导致用户沉浸感和信任度下降，还严重制约了数字人技术在教育、医疗、政务、金融等垂直领域的规模化落地和可持续发展。

特别是在中国教育数字化转型的宏观政策背景下，生成式数字人技术的应用需求显得尤为迫切且战略性强。2023年中共中央、国务院正式印发《数字中国建设整体布局规划》，明确提出要"构建普惠便捷的数字社会""大力实施国家教育数字化战略行动"，要求加快完善国家智慧教育平台，推进数字技术与教育教学深度融合。该规划为虚拟数字人技术在教育领域的应用提供了顶层设计指引。2024-2025年，教育部相继发布《国家智慧教育公共服务平台建设方案》《关于推进教育数字化转型的指导意见》以及2025年世界数字教育大会成果汇编，进一步强调要通过虚拟数字人技术实现"一人一课""因材施教"的个性化教学新模式。疫情后在线教育规模持续激增，据教育部2025年最新统计数据，国家智慧教育平台注册用户总量已突破1.78亿，日均访问量高达5200万次，总访问量累计超过726亿次，服务覆盖200多个国家和地区。广大一线教育工作者迫切需要将静态PPT课件、讲义材料快速转化为生动、专业的讲解视频，但传统人工录制方式耗时耗力、成本高昂，且难以实现个性化语音克隆、情感适配和多场景复用。

当前主流生成式数字人视频生成技术（如Wav2Lip [1]、SadTalker [2]、GeneFace [3]、GeneFace++ [4]、DIRFA等）虽已在实验室环境下取得令人瞩目的效果，但在真实教育场景中仍存在显著痛点：一是唇形同步精度不足，导致"口型不对嘴"的视觉失真，直接影响教学内容的专业性和可信度；二是面部表情自然度欠佳，数字人讲解过程显得机械僵硬，难以有效激发学生的学习兴趣和情感共鸣；三是跨模态一致性较差，语音语调与面部微表情、头部姿态脱节，降低教学沉浸感和知识传递效率；四是整体感知质量缺乏客观、可量化的评价标准，用户无法高效判断生成结果优劣，更难以针对性迭代优化。这些瓶颈不仅制约了数字人技术在高标准教育场景（如MOOC大规模开放在线课程、双师课堂、虚拟仿真实验室、特殊教育辅助教学）的规模化落地，还可能引发师生对AI生成内容的信任危机，甚至影响教育公平与教学质量。

然而，传统视频质量评估方法（如PSNR、SSIM [36]等，详见第二章2.2节）主要侧重于编码失真与信号保真度层面，难以有效捕捉说话人脸视频特有的多维度感知属性。单一维度评估工具（如SyncNet [22]仅关注唇形同步）也无法满足教育场景下多模态联合评价的复杂需求 [5][6]。面对上述技术挑战与应用痛点，构建面向生成式数字人的多维度质量评价模型，并将其集成至智能授课视频生成系统，已成为该领域亟待解决的核心科学问题和工程难题。

本研究的理论意义主要体现在以下四个方面：首先，首次系统提出涵盖唇形同步、表情自然度、音频质量、跨模态一致性和整体感知质量的五维客观评估体系，通过多模态特征融合与深度时序建模技术，实现对生成视频感知质量的精准量化，为说话人脸视频质量评估领域提供一套可复现、可比较、可扩展的基准框架；其次，创新性地引入动态任务权重策略，有效解决了多任务学习中的梯度冲突与负迁移问题，丰富并深化了多模态多任务学习的理论方法体系；再次，将跨模态一致性作为核心评价维度，弥补了传统单模态评估方法的不足，为后续多模态生成任务的感知建模提供了新范式；最后，为生成式AI在教育领域的负责任应用提供质量保障理论参考，促进技术伦理、学术诚信与质量控制的深度融合。

实践意义更为突出且直接：（1）集成少样本语音克隆技术（GPT-SoVITS [34]）、双模式面部驱动模型（SadTalker [2]与Wav2Lip [1]）以及像素级视频增强算法，构建前后端分离的B/S架构智能授课视频生成系统，实现从PPT课件与人像照片输入到成品数字人教学视频输出的全流程自动化处理，显著降低教育工作者制作高质量教学视频的技术门槛和时间成本；（2）嵌入质量评价闭环机制，形成"生成—评估—建议—调整—再生成"的迭代优化流程，支持用户基于客观、多维度的评分反馈持续改进视频质量；（3）为在线教育、虚拟主播、数字图书馆、远程培训等场景提供一套可复制、可扩展、可部署的技术方案，推动教育资源普惠化与数字化转型；（4）响应国家《数字中国建设整体布局规划》 [14]和教育数字化战略行动 [15]，为生成式AI在教育领域的健康有序发展提供可操作的质量保障参考，促进技术创新与社会责任的平衡统一。

此外，本研究还具有广泛的社会经济意义。在"双减"政策与乡村振兴战略背景下，数字人教学视频可有效辐射农村薄弱学校和边远地区，实现优质师资的共享与均衡；在特殊教育场景（如听障学生唇读辅助、语言障碍康复训练）中，高质量唇同步评估与表情自然度评价尤为关键；从经济角度看，该系统有望大幅降低教育视频制作成本（单课时成本可降至传统录制的1/10以下），提升教学效率，并为数字内容产业培育新增长点。长远来看，本研究成果有望为2035年我国"数字化发展水平进入世界前列"的数字中国建设目标贡献坚实的技术支撑。

综上所述，本文针对生成式数字人质量评价模型研究及智能授课视频生成系统构建这一关键问题开展系统性研究，不仅填补了现有技术在教育应用场景中的系统性空白，还为生成式AI的健康发展提供了闭环优化范式，具有重要的理论创新价值、实践应用价值和社会经济意义。

## 1.2 国内外研究现状

### 1.2.1 国际研究现状

国际学术界和产业界对说话人脸视频生成技术的探索始于21世纪初，并经历了从特定说话人依赖到泛化模型、从2D图像合成到3D神经渲染、再到多模态扩散驱动的三个主要演进阶段（各阶段技术原理详见第二章2.1节）。早期工作（2017年前后）主要依赖卷积神经网络（CNN）进行唇部像素预测，泛化能力极弱。2020年，Prajwal K R等人 [1]提出的Wav2Lip模型首次实现了任意身份、任意语音驱动的高精度唇同步，成为该领域公认的经典基准。

进入2023年，技术范式转向3D感知与神经辐射场渲染。Zhang W等人 [2]提出的SadTalker模型从单张人像照片生成风格化说话视频，显著提升了生成视频的自然运动感和表情流畅性。Ye Z等人系列工作（GeneFace [3]及GeneFace++ [4]，2023）基于NeRF渲染框架，实现了泛化音频-唇同步与实时生成能力，在保真度、多样性和实时性方面达到当时国际领先水平。

2024年至2025年间，扩散模型与多模态大模型驱动成为主流趋势。Wu R等人提出的DIRFA框架实现了多样化的真实表情动画生成；Wang X等人提出的MF-ETalk模型进一步增强了情感表达的丰富度和一致性。Toshpulatov等人 [5]在《Expert Systems with Applications》发表的综述系统分析了从2D到NeRF的演进路径，并指出跨模态一致性仍是主要瓶颈。Bai等人 [6]在IEEE发布的综述进一步强调了数据集构建与评价指标的重要性。

在质量评价领域，国际研究经历了从单一指标向多模态感知评估的转型。Chen S等人 [16]提出针对动态3D数字人的无参考质量评估指标，首次融入几何畸变与时序一致性建模。Quignon N等人 [12]开发的THEval框架整合了八项精细化指标，覆盖唇同步、表情自然度、音频保真度与整体感知质量。Su M等人 [10]提出的基于多模态特征表示的质量评估框架在NTIRE 2025 XGC Quality Assessment Challenge中获得Talking Head赛道冠军 [11]，进一步验证了多模态特征融合在说话人脸质量评估中的有效性。

教育应用层面，Guo P等人 [13]在ACM会议论文中探讨了数字人技术在教育改革中的应用。Liu Q [20]在《Frontiers of Digital Education》期刊探讨了数字人技术增强录播课程的个性化交互机制，但多数研究仍停留在原型验证阶段，缺乏集成质量评价闭环的完整端到端系统。总体而言，国际研究在生成技术前沿性和理论深度上处于领先地位，但在质量评价的系统性、教育场景适配深度以及全流程自动化系统构建方面仍存在明显不足。

### 1.2.2 国内研究现状

我国虚拟数字人研究起步虽略晚于国际，但在国家战略强力驱动与政策密集支持下，已呈现出爆发式发展态势。2023年中共中央、国务院印发的《数字中国建设整体布局规划》[14]以及后续教育部系列文件 [15]，均将虚拟数字人明确列为教育数字化转型的关键支撑技术。宋一飞等人 [8]在《计算机辅助设计与图形学学报》发表的《数字说话人视频生成综述》，系统梳理了GAN、扩散模型与NeRF技术在中文语音场景下的适配问题，并重点分析了中文复杂韵律对唇同步精度的独特挑战。张冰源等人 [17]在《大数据》期刊发布的《数字说话人脸生成技术综述》，进一步对比分析了国内外数据集差异与评估策略，强调中文教育场景需额外关注情感表达与教学节奏适配。乐铮等人 [7]在《计算机研究与发展》发表的《音频驱动的说话人面部视频生成与鉴别综述》，对比分析了经典算法与最新成果，并提出适用于中文语音的鉴别与优化方法。李青等人（2024）在《现代远距离教育》期刊发表的《教育虚拟数字人标准体系设计及其路径规划》，首次构建了涵盖功能框架、技术规范、质量评价、安全伦理的完整标准体系，为教育领域虚拟数字人规范落地提供了政策与技术双重支撑。该文明确指出，教育虚拟数字人需同时满足表达能力、教学能力和用户体验三维质量评估要求。

SenseTime等企业主导的《信息技术 面向客服的虚拟数字人通用技术要求》（GB/T 46483-2025）[19]国家标准于2025年正式发布，进一步规范了数字人在教育等领域的应用。Yu Q等人 [18]在国际会议论文中系统总结了语音驱动说话数字人视频生成方法，讨论了GAN、扩散模型和NeRF的关键技术与数据集。

然而，国内质量评价研究相对滞后。大多数工作仍聚焦于单一维度评估（如唇同步精度或视觉保真度），缺乏多任务联合、多模态深度融合的客观评价体系。现有智能授课系统多为功能演示原型，鲜有嵌入实时质量评价闭环的端到端解决方案。在教育特定优化方面（如教学节奏适配、表情情感表达与学生认知匹配），与国际多模态感知评价框架相比仍存在一定差距。

综上所述，国内外研究虽已取得丰硕成果，但仍存在三大明显空白：（1）评价维度单一，无法全面反映教育场景下复杂的感知需求；（2）生成系统缺乏客观反馈与迭代优化机制，用户难以高效提升视频质量；（3）教育应用适配不足，缺少支持PPT课件自动解析、人像驱动与课件合成的全自动化管线。本文正是针对上述研究空白，提出基于多模态融合Transformer的多任务质量评价模型，并构建集成质量闭环的智能授课视频生成系统，旨在填补国内研究在教育垂直场景的系统性空白，为数字人技术在我国教育领域的规模化应用提供坚实技术支撑。

## 1.3 研究内容与结构安排

本研究的核心内容主要包括以下三个相互关联、层层递进的方面：

第一，构建基于多模态融合Transformer的多任务质量评价模型。针对多模态异构特征融合、多任务梯度冲突以及高维特征空间下的泛化难题，本文提出完整的端到端框架，联合提取视觉、音频、面部关键点及辅助几何表情编码等多模态特征，实现唇形同步、表情自然度、音频质量、跨模态一致性与整体感知质量五维质量分数的联合回归预测。通过特征标准化、PCA降维、模态嵌入对齐以及动态任务权重策略，有效缓解了尺度失衡、维度灾难与梯度冲突问题（技术细节详见第二章与第三章）。

第二，集成前沿生成与增强技术，搭建前后端分离的B/S架构智能授课视频生成系统。本文融合GPT-SoVITS [34]少样本语音克隆、SadTalker [2]三维感知面部驱动、Wav2Lip [1]高精度唇同步驱动、像素级面部增强与背景透明化处理等技术，实现从用户上传PPT课件与人像照片到最终数字人教学视频输出的全流程自动化处理。系统采用Vue 3前端框架与Flask后端框架，支持异步任务调度、多用户并发隔离与实时进度反馈，显著提升了系统的可用性与可扩展性。

第三，嵌入质量评价闭环优化机制。在视频生成完成后，系统自动调用所构建的多维度质量评价模型进行实时评分，输出包含总分、等级、分项得分、评测结论以及针对性参数优化建议的结构化报告，形成"生成—评估—建议—调整—再生成"的完整迭代优化闭环。该机制使用户能够基于客观数据持续改进生成参数，最终产出满足教育教学需求的高质量数字人视频。

为验证上述研究内容的有效性，本文基于CH-SIMS v2扩展数据集开展大规模实验，验证了质量评价模型与主观感知的高度一致性；同时通过系统功能测试、多用户并发测试与端到端流程测试，验证了生成系统的稳定性、实用性与可扩展性。

论文结构安排如下：第一章为绪论，系统阐述研究背景、意义、国内外研究现状以及本文主要研究内容与结构安排；第二章综述说话人脸视频生成技术、数字人质量评价方法、多模态融合理论、多任务学习策略以及系统架构相关技术基础；第三章详细设计并实现多模态融合Transformer的多任务质量评价模型算法；第四章开展数据集构建、消融实验与性能分析；第五章介绍智能授课视频生成系统的架构设计、关键模块实现与全面功能测试；第六章总结全文工作，提炼主要创新点，并对未来研究方向进行展望。

通过上述系统性研究，本文旨在为生成式数字人技术在教育领域的应用提供完整的技术解决方案，推动我国教育数字化转型向更高质量、更普惠的方向发展。

# 2 相关技术基础与理论框架

## 2.1 说话人脸视频生成技术

说话人脸视频生成（Talking Face Generation）技术旨在通过音频信号驱动静态人脸图像，合成具有真实唇形同步与自然表情的动态视频。根据技术路线的演进，现有方法主要经历了基于生成对抗网络（GAN）[21]的2D方法、基于神经辐射场（NeRF）的3D感知方法，以及基于扩散模型（Diffusion Model）和Transformer [41]的新一代方法三个发展阶段。不同技术路线在生成质量、计算效率、泛化能力等方面各具优势，为构建实用的数字人系统提供了多样化的技术选型。

### 2.1.1 基于GAN的方法与Wav2Lip架构详解

生成对抗网络（GAN）[21]作为早期说话人脸生成的主流范式，通过生成器与判别器的对抗训练机制提升合成质量。生成器负责从输入条件（音频特征、参考图像）生成逼真的面部图像，而判别器则试图区分真实样本与生成样本，二者在对抗过程中不断优化，最终使生成器能够产生难以区分真伪的高质量图像。

Wav2Lip模型 [1]是GAN范式在说话人脸生成领域的里程碑工作，其核心创新在于引入了预训练的唇同步专家网络（SyncNet）[22]作为判别器，强制生成器产生与音频时序精确对齐的唇形运动。与传统方法依赖特定说话人数据训练不同，Wav2Lip通过大规模野外视频（In-the-Wild）预训练，实现了任意身份、任意语音的高精度唇同步，在跨身份生成场景下表现出优异的泛化能力。

**Wav2Lip技术架构**采用经典的编码器-解码器结构，其核心设计如图2.1所示，主要包含三个关键组件：

![图2.1 Wav2Lip网络架构图](media/wav2lip_architecture.png)
**图2.1 Wav2Lip网络架构图**：该架构包含双分支生成器结构。左分支为身份编码器，输入参考帧（Reference Frame）与掩码帧（Masked Frame）提取身份特征；右分支为音频编码器，处理梅尔频谱（Mel-Spectrogram）特征。两分支特征在解码器中通过注意力机制融合，输出生成的口型区域，最终通过泊松融合 [64]与原始图像合成。

具体而言，**生成器（Generator）**采用U-Net架构，包含下采样编码器与上采样解码器。编码器接收两个视觉输入：一是提供身份信息的参考帧，通常从视频中随机选取以确保身份一致性；二是提供头部姿态与背景上下文的掩码帧（嘴部区域被遮盖，仅保留上半面部和背景）。这种双输入机制确保生成器在改变口型的同时保持原始身份和姿态不变。音频分支以连续梅尔频谱为输入，通过一维卷积层逐步提取时序特征，捕捉语音的韵律和发音特征。

为实现音频-视觉时序对应，Wav2Lip采用注意力机制进行特征融合。视觉特征与音频特征在通道维度拼接后，通过卷积层学习两者的非线性映射关系，解码器通过转置卷积逐步恢复空间分辨率，最终输出生成的嘴部区域RGB图像。这种注意力融合机制能够有效对齐音频节奏与视觉口型变化，确保生成的唇部动作与语音内容同步。

**判别器（Discriminator）**采用双判别器策略。**视觉质量判别器**使用PatchGAN结构，关注生成图像的局部纹理真实性，确保生成的嘴部区域与周围皮肤在肤色、纹理上无缝衔接，避免出现明显的拼接痕迹。**唇同步判别器**基于预训练的SyncNet网络 [22]，通过对比学习判断音视频同步性。SyncNet将视频帧和音频片段映射到嵌入空间，通过计算两者的距离来判断同步性：距离越小表示同步性越好，距离越大则表示不同步。在Wav2Lip中，SyncNet参数被冻结，其输出的唇同步置信度作为监督信号，强制生成器产生与音频精确对齐的口型运动。

训练过程采用**两阶段策略**：第一阶段在配对数据上进行监督学习，优化重建损失，确保生成图像与目标图像的像素级相似性；第二阶段引入SyncNet置信度损失，通过强化学习思路微调生成器，确保唇形与上下文时序连贯，提升长序列生成的稳定性。

在此基础上，IP-LAP提出了基于中间表示的两阶段生成策略，首先通过音频到面部关键点的映射生成稀疏结构引导，再利用关键点引导视频生成网络合成最终图像。该方法通过显式的几何约束减少了直接像素预测的不确定性，提升了生成视频的稳定性。CodeTalker [24]则采用离散化语音表示学习，结合向量量化变分自编码器（VQ-VAE）将连续音频特征离散化为码本索引，再通过自回归模型预测面部动作离散编码，实现了语音语义的高效编码与面部动作精确解码。然而，GAN-based方法在处理长时序依赖时存在身份漂移现象，即随着生成时间的延长，人物面部特征可能逐渐失真，且对抗训练的不稳定性限制了生成表情的自然度。

### 2.1.2 基于3D感知的方法与SadTalker架构详解

神经辐射场（NeRF）与3D可形变模型（3D Morphable Model, 3DMM）[27]的引入为说话人脸生成带来了三维感知能力，有效解决了2D方法中头部姿态僵硬和身份漂移问题。3DMM通过主成分分析（PCA）[44]将面部几何与纹理表示为身份、表情等系数的线性组合，能够精确建模面部肌肉运动的几何变化。神经辐射场则通过隐式神经表示建模场景的三维几何与外观，能够从任意视角渲染逼真的面部图像。

AD-NeRF [26]作为首个将NeRF应用于说话人头合成的代表性工作，采用音频驱动的隐式神经表示分别建模头部与躯干区域。其神经辐射场将三维空间坐标与视角方向映射为颜色值和体积密度，通过体渲染技术生成最终图像，实现了端到端可微分渲染。这种方法能够生成具有真实光照效果和视角变化的说话人脸视频，但计算开销较大，难以满足实时应用需求。

**SadTalker技术架构** [2]巧妙地结合了3DMM的显式几何表示与隐式神经渲染优势，通过显式的三维运动系数作为中间桥梁，有效平衡了生成质量与计算效率。其核心流程如图2.2所示，包含三个主要模块：

![图2.2 SadTalker系统架构图](media/sadtalker_architecture.png)
**图2.2 SadTalker系统架构图**：系统从单张肖像中提取身份系数。通过并行的ExpNet（表情网络）与PoseVAE（姿态VAE）分别生成表情系数与头部姿态序列。最终通过3D感知面部渲染器将3D运动系数映射回图像空间，输出生成视频。

**ExpNet（Expression Network）**基于条件变分自编码器（CVAE）架构，以音频特征为条件学习从梅尔频谱到3DMM表情系数的映射。CVAE通过引入潜在变量建模数据分布，能够生成多样化的表情序列而非确定性的单一输出。为解决表情多样性与身份保持的矛盾，ExpNet引入了蒸馏的音频-表情特征技术：通过预训练的表情识别网络提取音频对应的参考表情作为先验约束，确保生成表情符合语音情感。同时采用身份感知生成策略，将身份系数作为条件输入，确保生成表情符合特定人物的肌肉运动规律，避免跨身份失真。

**PoseVAE**负责生成自然的头部姿态序列。与表情不同，头部姿态具有更强的随机性和长程依赖性。PoseVAE通过编码器将真实姿态序列压缩到潜在空间，学习姿态的分布特征；推理阶段从先验分布采样，通过解码器生成与音频节奏匹配的头部姿态序列。这种概率建模方式避免了确定性生成带来的重复性动作，使头部运动更符合人类说话时的自然习惯，如强调时的点头、思考时的侧头等。

**3D感知面部渲染器**连接3D运动系数与2D图像。给定表情系数、姿态系数与身份系数，渲染器首先通过3DMM [27]重建三维面部网格，然后利用薄板样条插值（TPS）对参考图像进行空间变换，生成与目标姿态和表情对齐的面部图像。TPS插值通过计算参考点与目标点之间的变形场，实现平滑的几何变换，保持面部纹理的自然过渡。

SadTalker还引入GFPGAN [28]进行面部细节增强，修复渲染过程中可能出现的模糊与伪影，支持高分辨率视频生成。这种"生成+增强"的两阶段策略显著提升了视觉质量，使生成的视频在细节丰富度和真实感方面接近专业录制水平。

GeneFace系列 [3][4]进一步推动了该领域发展。GeneFace设计了基于变分推断的音频到面部运动系数映射网络，通过引入变分运动生成器学习音频与3DMM系数的概率映射关系，结合高效神经渲染器解决了长序列时序一致性问题。GeneFace++ [4]通过引入基于扩散模型的运动生成器与实时渲染管线，在保持质量的同时显著提升了推理速度，达到了实时生成的性能要求。ER-NeRF [29]提出了区域感知的三平面哈希编码机制，针对唇部区域精细建模，有效提升了口型同步精度。

### 2.1.3 扩散模型、Transformer与少样本生成技术

近年来，扩散概率模型和基于Transformer [41]的架构逐渐成为说话人脸生成领域的主流方向。扩散模型通过逐步去噪的过程生成数据，在生成质量和多样性方面展现出超越GAN的潜力。其基本思想是定义一个前向过程逐步向数据添加噪声，然后训练神经网络学习反向的去噪过程，从而从纯噪声中恢复出清晰的图像或视频。

FaceDiffuser [30]将扩散模型应用于3D面部动画生成，通过条件扩散过程建模音频到面部网格顶点的映射，在顶点级误差和面部动态偏差指标上取得最优性能。DiffTalk [31]提出了基于扩散的音频驱动框架，通过解耦身份、表情和姿态表示实现精细控制，采用Classifier-free Guidance技术增强音频条件的控制力，使生成结果更加稳定可控。

FaceFormer [32]采用基于自注意力的编码器-解码器结构，直接建模音频序列到3D面部顶点序列的映射。Transformer架构 [41]通过多头注意力机制捕捉音频序列中的长程依赖关系，能够建立音频特征与面部动作之间的复杂映射，避免了3DMM参数的信息瓶颈。CodeTalker [24]结合离散表示学习与Transformer，通过向量量化将连续音频特征量化为离散码本索引，再通过自回归Transformer预测面部动作离散编码，实现了高质量的语音驱动动画。

在少样本与跨身份生成方面，NeRFFaceSpeech [33]利用预训练生成模型构建3D感知面部特征空间，通过射线变形技术实现单样本音频驱动合成。GPT-SoVITS [34]代表了当前少样本语音合成的前沿水平，结合GPT模型的语义建模与SoVITS的声学建模能力，仅需约1分钟参考音频即可实现高质量音色克隆与跨语言语音合成。MuseTalk [35]通过时空采样机制实现了实时高保真视频配音，利用空间注意力机制定位唇部区域，时间注意力机制保证时序连贯性，为实时通信场景提供了可行方案。

![图2.3 少样本语音克隆技术流程图](media/few_shot_voice_cloning.png)
**图2.3 少样本语音克隆技术流程图**：该图展示了GPT-SoVITS [34]的两阶段架构。第一阶段（SoVITS模块）接收参考音频，通过声学编码器提取音色特征，结合文本输入训练声学模型；第二阶段（GPT模块）接收目标文本，预测语义token序列，再通过声码器生成最终波形。这种分离架构实现了音色与内容的解耦，支持跨语言音色迁移。

## 2.2 质量评估与多模态分析理论

### 2.2.1 数字人视频质量评估体系

随着生成技术的快速发展，如何客观、全面评估生成式数字人视频的感知质量成为亟待解决的关键问题。传统视频质量评估方法主要针对压缩失真或传输损伤，难以有效捕捉说话人脸视频特有的多维度感知属性，如唇同步精度、表情自然度、时序连贯性等。评估体系经历了从像素级到感知级，再到多维度语义级的发展过程。

**传统像素级与感知质量指标**方面，峰值信噪比（PSNR）通过计算均方误差来量化失真程度，虽然计算简便，但其基于像素级差异的度量方式与人眼感知存在显著偏差，尤其在评估生成式内容时，高PSNR值并不必然对应高感知质量。结构相似性指数（SSIM）[36]通过比较亮度、对比度和结构信息的相似性，在一定程度上更符合人类视觉系统特性，但对于说话人脸视频这类具有复杂时序动态的内容，静态的SSIM指标难以捕捉面部表情变化的流畅性与自然度。

Fréchet Inception Distance（FID）[37]作为生成模型评估的主流指标，通过计算生成图像与真实图像在深度特征空间中的分布距离，衡量生成样本的真实感与多样性。FID值越低，表明生成图像的分布越接近真实图像分布。然而，FID主要关注单帧图像质量，对时序一致性和音视频同步性缺乏直接度量。

![图2.4 多维度质量评估体系框架图](media/quality_assessment_framework.png)
**图2.4 多维度质量评估体系框架图**：该图展示了从输入视频到多维评分的处理流程。视频流经过视觉分析、音频分析、几何分析三个并行分支，分别提取视觉质量特征、音频质量特征、面部关键点特征，再通过多模态融合模块生成唇同步、表情自然度、音频质量、跨模态一致性、整体感知质量五个维度的评分。

**唇同步精度评估指标**是说话人脸视频最核心的质量维度，直接关系到生成内容的可信度与可懂度。SyncNet [22]作为该领域最具影响力的自动化评估工具，通过训练孪生网络学习音频与视频帧的嵌入表示，利用对比损失使得同步的音视频对在嵌入空间中距离相近，而不同步的配对距离较远。基于SyncNet，研究者提出了两个互补的评估指标：Lip Sync Error - Confidence（LSE-C）反映模型对音视频同步性的置信度，值越高表示同步性越好；Lip Sync Error - Distance（LSE-D）则直接度量音频特征与视频特征之间的距离，值越低表示对齐精度越高。

Landmark Distance（LMD）从几何角度评估唇同步质量，通过比较生成视频与参考视频中面部关键点的位置偏差来量化口型准确性。Mouth Landmark Distance（M-LMD）专注于口周区域的关键点匹配，而Full-face Landmark Distance（F-LMD）则评估整个面部结构的保持程度。这些几何指标能够精确量化口型开合幅度、嘴角移动轨迹等细节，为唇同步评估提供了客观的物理度量。

**多维度感知质量评估框架**近年来逐渐受到重视。THEval框架 [12]提出了涵盖八项精细化指标的评估体系，包括唇同步精度、表情自然度、音频保真度、时序一致性、身份保持度、视觉质量、整体感知质量和跨模态协调性。Su等人 [10]提出的基于多模态特征表示的质量评估方法在NTIRE 2025挑战赛 [11]中取得领先性能，融合视觉、音频与几何线索，通过端到端多任务学习框架实现精准预测。Chen等人 [16]针对动态3D数字人提出了无参考质量评估指标，结合几何畸变分析与时序一致性建模，有效解决了缺乏参考视频时的评价难题。

主观评估方面，Mean Opinion Score（MOS）测试仍然是衡量感知质量的金标准。通过组织大量受试者对生成视频的自然度、同步性、整体质量等维度进行评分，可以建立客观指标与主观感知之间的映射关系。近期研究表明，结合眼动追踪技术的细粒度主观评估 [39]能够更准确捕捉人类对面部异常区域的敏感度，为客观指标设计提供生理学依据。

### 2.2.2 多模态特征融合技术

多模态融合是处理异构数据（视觉、音频、文本等）的关键技术，在说话人脸视频质量评估中起着核心作用。有效的多模态融合能够挖掘跨模态的互补信息，提升模型对音视频协调性的理解能力。根据融合发生的位置，多模态融合策略可分为早期融合、晚期融合和模型级融合。早期融合在特征提取阶段即进行模态拼接，虽然能够保留细粒度的原始信息，但面临维度灾难和模态间语义鸿沟的挑战。晚期融合则先在各模态独立进行预测，再通过加权平均或投票机制整合结果，实现简单但忽略了模态间的内在关联，难以捕捉跨模态的协同效应。

Transformer架构 [41]为模型级融合提供了强大工具。Cross-Attention机制允许一个模态的Query去关注另一个模态的Key和Value，从而实现信息的跨模态传递与增强。这种机制能够自适应地捕捉视觉区域与音频片段之间的对应关系，例如将唇部视觉特征与对应时间窗口的音频特征进行关联，从而更准确地评估唇同步质量。

![图2.5 多模态特征融合架构对比图](media/multimodal_fusion_comparison.png)
**图2.5 多模态特征融合架构对比图**：左侧展示早期融合策略，在特征提取层直接拼接各模态特征；中间展示晚期融合策略，各模态独立预测后决策层融合；右侧展示基于Transformer的模型级融合，通过Cross-Attention机制实现深层语义交互。图中箭头粗细表示信息流强度，虚线表示可选的残差连接。

MutualFormer [42]提出的Cross-Diffusion Attention（CDA）机制通过定义基于个体模态亲和力的交叉亲和力，有效避免了传统Cross-Attention中的模态鸿沟问题。该方法首先计算模态内的自相似度矩阵，然后通过扩散过程增强跨模态关联，在RGB-Depth显著性检测等任务中展现出优越性能。TACFN [43]针对跨模态注意力中的特征冗余问题，提出了自适应跨模态融合块，首先通过自注意力进行模态内特征选择，筛选出对另一模态最有用的信息再进行交互，显著提升了融合效率。

模态对齐与语义统一方面，由于不同模态的数据具有异构的数值分布和语义空间，直接融合往往效果不佳。Z-score标准化通过对每类模态特征独立标准化消除尺度差异，使不同模态的特征具有 comparable 的数值范围。PCA降维 [44]等预处理手段能够缓解高维特征带来的计算压力，通过保留主成分去除冗余信息，提升特征的信噪比。

对比学习在跨模态对齐中展现出巨大潜力。通过构造正样本对（同步的音视频片段）和负样本对（错配或不同步的片段），模型能够学习到具有判别性的跨模态表示。CLIP等预训练模型展示了大尺度对比预训练在建立视觉-语言对齐方面的有效性，类似的思路也被应用于音视频对齐任务，通过最大化同步片段的互信息来提升表征质量 [45]。

### 2.2.3 多任务学习优化策略

多任务学习（Multi-Task Learning, MTL）[46]旨在通过共享表示同时学习多个相关任务，提升数据效率和模型泛化能力。在说话人脸视频质量评估中，同时预测唇同步、表情自然度、音频质量等多维度评分天然构成多任务学习问题。通过共享底层特征表示，模型能够捕捉不同质量维度之间的内在关联，例如唇同步质量与整体感知质量的相关性，从而提升各任务的预测精度。

硬参数共享与软参数共享是两种主要架构。硬参数共享在底层网络共享全部参数，顶层分离为任务特定的分支，这种结构简单高效但容易导致任务间的负迁移，即一个任务的优化改善以牺牲其他任务性能为代价。软参数共享为每个任务保留独立的网络，通过正则化约束促使不同任务的参数相似，或者通过门控机制动态控制信息共享的程度。Cross-stitch网络 [47]通过可学习的线性组合实现任务间信息共享程度的自适应调节，是软参数共享的代表性方法。

任务关系学习与不确定性加权方面，当各任务的最优梯度更新方向在参数空间中可能存在冲突时，需要设计自适应的动态任务权重策略。GradNorm算法 [48]通过动态调整各任务的权重，使不同任务的梯度范数保持平衡，从而协调训练进度。该算法监控各任务的梯度范数，对训练较快的任务降低权重，对训练较慢的任务增加权重，确保各任务以相似的速度收敛。

![图2.6 多任务学习权重动态调整示意图](media/multitask_weight_adjustment.png)
**图2.6 多任务学习权重动态调整示意图**：左侧展示固定权重策略，各任务损失简单相加，导致简单任务主导梯度；右侧展示动态权重策略，根据任务难度和不确定性自动调整权重系数。图中饼图大小表示权重分配，箭头粗细表示梯度贡献程度。

Kendall等人 [49]提出的基于同方差不确定性的任务加权策略为多任务优化提供了理论框架。该方法将每个任务的损失乘以其对应的不确定性，通过最大化似然函数自动学习最优权重。不确定性高的任务（难任务）获得较低权重，不确定性低的任务（易任务）获得较高权重，从而实现了对任务难度的自适应平衡。这种策略无需手动调整权重，且能够根据训练过程中的不确定性变化动态调整，有效缓解了多任务学习中的梯度冲突问题。

动态权重平均（DWA）策略通过监控各任务损失的相对变化率来调整权重，使学习速度较慢的任务获得更高关注。与基于不确定性的方法相比，DWA不需要额外的参数估计，实现更为简便。MetaBalance等元学习方法则通过引入辅助的元网络学习任务间的最优组合策略，能够根据当前模型状态动态调整权重，在处理复杂任务关系时表现出更强的适应性。

梯度冲突与帕累托最优方面，当多个任务的梯度在参数空间中方向冲突时，简单的加权平均可能导致优化震荡或收敛到次优解。PCGrad算法 [50]通过将各任务的梯度投影到彼此正交的方向，消除了梯度冲突。帕累托多任务学习 [51]将多目标优化引入深度学习，寻找帕累托前沿上的最优解集，使得在不损害任何任务性能的前提下无法进一步提升其他任务性能。这种方法虽然计算复杂度高，但能够从理论上保证任务间的最优平衡。

## 2.3 语音合成与系统架构技术

### 2.3.1 少样本语音克隆与情感控制技术

语音合成（Text-to-Speech, TTS）技术是数字人系统的核心组件之一，负责将文本内容转换为具有特定音色的自然语音。传统TTS系统通常采用级联架构，包括文本分析、声学模型和声码器三个模块。文本前端将输入文本转换为语言学特征（如音素序列），声学模型预测声学特征（如梅尔频谱），声码器最终合成波形。这种流水线结构复杂且误差会逐级累积，且各模块独立优化难以达到全局最优。

端到端神经TTS模型通过编码器-解码器架构直接从字符或音素序列生成梅尔频谱，简化了流程并提升了自然度。Tacotron [52]采用基于注意力机制的序列到序列模型，通过注意力权重动态选择编码器隐藏状态，建立文本与语音之间的对齐关系。FastSpeech则通过非自回归并行生成和长度调节器显著提升了推理速度，实现了实时语音合成。

VITS [53]进一步将声学模型与声码器整合，通过变分自编码器和对抗训练实现了高质量的单阶段语音合成。其目标函数包含重构损失、KL散度和对抗损失，通过变分推断学习潜在的声音表征，同时利用对抗训练提升合成语音的自然度。

![图2.7 语音合成技术演进路线图](media/tts_evolution.png)
**图2.7 语音合成技术演进路线图**：从左至右展示从传统级联TTS（多模块串联）到端到端神经TTS（编码器-解码器架构）再到少样本语音克隆（预训练+微调范式）的技术演进。图中用不同颜色区分文本处理、声学建模、波形生成三个阶段，箭头粗细表示信息流强度。

少样本语音克隆旨在仅通过极少量的目标说话人参考音频（通常几秒到几分钟）克隆其音色特征。SV2TTS框架 [54]引入说话人嵌入机制，通过预训练的说话人验证网络提取目标音色嵌入，指导TTS模型生成具有目标音色的语音。该嵌入向量通过广义端到端损失训练得到，能够有效区分不同说话人，实现跨音色迁移。

GPT-SoVITS [34]代表了当前开源社区中最先进的少样本语音克隆方案。其核心创新在于分离语义建模与声学建模：首先使用GPT模型预测语音的语义token，捕捉内容信息和韵律特征；然后通过基于VITS [53]改进的SoVITS模块将语义token转换为声学特征。这种架构不仅降低了对训练数据量的要求，还实现了跨语言音色迁移。其训练过程包括两个阶段：第一阶段训练SoVITS模块进行声学特征重建，学习目标说话人的音色特征；第二阶段训练GPT模型进行语义token预测，建立文本到语义的映射关系。

Neural Audio Codec通过神经编解码技术学习离散的音频表示，结合语言模型进行自回归生成，在保持高音质的同时实现了对语音风格的精细控制。SoundStream [55]采用残差向量量化技术，将音频压缩为离散token，大幅降低存储和传输开销，同时支持高质量的音频重建。

情感语音与风格控制方面，教育场景中的数字人不仅需要准确的语音内容，还需要匹配教学情境的情感表达（如鼓励、严肃、亲切等）。StyleTTS [56]等模型通过引入风格预测器，从参考音频中提取韵律特征（如音高、能量、时长），实现了对合成语音风格的可控调节。其风格向量通过参考音频编码器提取，然后通过自适应实例归一化注入到解码器中，实现风格迁移。

Emotional Voice Conversion（EVC）技术能够在保持内容不变的前提下，将中性语音转换为具有特定情感的语音，这对于后期调整数字人教学视频的情感基调具有重要意义。基于CycleGAN或V-StarGAN的EVC方法通过学习情感无关的内容表示与情感特定的风格表示的解耦，实现了灵活的情感迁移。这些方法通过对抗损失和循环一致性损失学习双向映射，确保转换后的语音保持内容一致性同时具有目标情感特征。

### 2.3.2 前后端分离架构与工程优化技术

构建实用的数字人教学视频生成系统不仅需要先进的算法模型，还需要合理的系统架构设计与工程优化，以确保系统的可用性、可扩展性和实时性。

现代Web应用普遍采用前后端分离架构 [58][59]，前端负责用户界面渲染与交互逻辑，后端专注业务逻辑与计算密集型任务。这种架构通过RESTful API [58]进行通信，实现了关注点分离，便于团队并行开发和独立部署。RESTful API遵循Representational State Transfer原则，使用HTTP方法对应资源的增删改查操作，具有无状态、可缓存、分层系统等特性，非常适合机器学习模型的服务化部署。

![图2.8 前后端分离系统架构图](media/system_architecture.png)
**图2.8 前后端分离系统架构图**：该图展示典型的B/S架构数字人视频生成系统。前端层（Vue 3）负责用户交互、状态管理、进度展示；API网关层负责请求路由、负载均衡、认证授权；微服务层包含语音合成服务、面部驱动服务、视频合成服务等；任务队列层（Redis/RabbitMQ）处理异步任务调度；模型推理层（GPU集群）执行深度学习推理；存储层管理视频、音频、模型权重等资源。虚线表示异步消息流，实线表示同步HTTP请求。

前端框架方面，Vue 3、React等现代JavaScript框架提供了组件化开发模式和响应式状态管理，配合TypeScript能够提供良好的类型安全与开发体验。Vue 3引入了Composition API，通过setup函数和响应式引用更好地组织逻辑代码，提升代码复用性。Pinia或Vuex等状态管理库能够有效管理复杂的应用状态，特别是在处理异步任务状态（如视频生成进度）时至关重要。Pinia采用Store模式，通过state、getters、actions组织状态管理，支持TypeScript类型推断，便于大型项目的维护。

后端方面，Python Flask或FastAPI等轻量级框架适合快速构建机器学习模型的服务接口。FastAPI基于Starlette和Pydantic，支持异步编程和自动数据验证，性能接近Node.js和Go。异步任务队列（如Celery、RQ）通过消息中间件（Redis、RabbitMQ）将耗时操作（如深度学习推理）与主应用解耦，避免请求阻塞，提升系统并发处理能力。Celery采用生产者-消费者模式，任务发布到Broker，由Worker进程异步执行，结果存储在Backend，确保系统的响应性和可扩展性。

模型部署与推理优化方面，深度学习模型面临计算资源消耗大、推理延迟高等挑战。模型量化通过将浮点参数转换为低精度整数（如INT8），在轻微损失精度的情况下显著减少模型体积和计算量。Post-Training Quantization无需重新训练，而Quantization-Aware Training在训练中模拟量化误差，精度更高。知识蒸馏 [61]通过训练轻量级学生模型模仿教师模型的行为，实现模型压缩与加速，蒸馏过程利用教师模型的软目标指导学生模型学习，保留更多的信息。

ONNX格式提供了跨框架的模型表示标准，使得在不同硬件平台（如GPU、TPU、CPU）上的部署更加灵活。ONNX Runtime通过图优化和算子融合提升推理速度。TensorRT [62]等推理优化库通过内核自动调优、精度校准等技术，能够进一步提升GPU上的推理吞吐量。对于视频生成这类计算密集型任务，采用流式处理或分块处理策略，将长视频分割为短片段并行处理，再拼接为完整视频，能够有效降低单任务的内存占用并提升整体处理速度。

多媒体处理与视频编码方面，FFmpeg [63]作为开源多媒体处理框架，提供了音视频的编解码、格式转换、滤镜处理等完整功能，是数字人视频合成管线的基础工具。FFmpeg支持多种编码标准，包括H.264/AVC、H.265/HEVC、VP9、AV1等。H.265/HEVC相比H.264在相同码率下提供更好的画质，但编码复杂度更高；AV1作为开源免专利费编码器，在压缩效率上具有优势。

在数字人视频合成中，Alpha通道合成技术用于将透明背景的数字人前景层与课件背景层融合。Alpha合成根据前景透明度混合前景和背景颜色，实现自然的叠加效果。图像融合算法需要考虑边缘平滑、颜色一致性等因素，以避免出现明显的拼接痕迹。泊松融合 [64]通过求解泊松方程实现无缝拼接，保持源图像的梯度信息同时匹配目标图像的边界条件。多频段混合则通过拉普拉斯金字塔分解，在不同频率层分别融合，保留细节的同时平滑过渡，适用于分辨率较高的视频合成。

实时通信场景下，WebRTC [65]技术提供了浏览器端的双向音视频传输能力，结合MediaStream API能够实现数字人视频的实时推流与展示。WebRTC采用SRTP传输媒体数据，通过ICE框架实现NAT穿越，支持P2P直连和中继转发两种模式，确保在不同网络环境下的连通性。

## 2.4 本章小结

本章系统梳理了构建生成式数字人质量评价模型与智能授课视频生成系统所需的核心理论与技术基础。在说话人脸生成技术方面，详细解析了基于GAN的Wav2Lip架构 [1]（编码器-解码器结构结合SyncNet [22]监督实现高精度唇同步）与基于3D感知的SadTalker架构 [2]（通过ExpNet和PoseVAE分别建模表情与姿态，平衡计算效率与生成自然度），深入探讨了扩散模型、Transformer [41]及少样本生成技术的最新进展，包括扩散概率模型的基本原理、Transformer的注意力机制、以及GPT-SoVITS [34]的语义-声学分离架构。

在质量评估与多模态分析方面，阐述了从传统像素级指标（PSNR、SSIM [36]）到感知指标（FID [37]）、唇同步指标（LSE-C/D）再到多维度评估框架（THEval [12]）的发展脉络，详细描述了各类指标的物理意义与应用场景；分析了基于Transformer的多模态融合技术（Cross-Attention、CDA机制 [42]）与多任务学习优化策略（不确定性加权 [49]、梯度冲突解决 [50]），包括同方差不确定性加权的基本原理、PCGrad [50]的梯度投影思路等。

在语音合成与系统架构方面，探讨了从传统级联TTS到端到端神经TTS、再到GPT-SoVITS [34]少样本语音克隆的技术演进，详细描述了VAE、GAN [21]在语音合成中的应用，以及StyleTTS [56]的风格控制机制；深入分析了前后端分离架构 [58][59]、异步任务调度、模型量化与知识蒸馏 [61]等工程实践要点，包括系统组件划分、数据流向、性能优化策略等。

综上所述，现有研究为本文工作奠定了坚实基础，但在教育场景适配、多维度质量评估体系构建、以及生成-评价闭环系统实现等方面仍存在明显不足。特别是现有质量评估方法多聚焦于单一维度，缺乏对跨模态一致性的系统性建模；生成系统鲜有集成实时质量反馈与迭代优化机制。下一章将详细阐述本文提出的多模态融合Transformer多任务质量评价模型的具体设计方案，以及集成该模型的智能授课视频生成系统的构建方法。

---

**参考文献**

[1] Prajwal K R, Mukhopadhyay R, Namboodiri V P, et al. A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 484-492.

[2] Zhang W, Cun X, Wang X, et al. SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 8652-8661.

[3] Ye Z, Jiang Z, Ren Y, et al. GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis[C]//The Twelfth International Conference on Learning Representations. 2023.

[4] Ye Z, He J, Jiang Z, et al. GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation[J]. arXiv preprint arXiv:2305.00787, 2023.

[5] Toshpulatov M, Lee W, Lee S. Talking human face generation: A survey[J]. Expert Systems with Applications, 2023, 219: 119678.

[6] Bai X, et al. A Survey on Audio-Driven Talking Face Generation[J]. IEEE Access, 2025.

[7] Le Zheng, Hu Yongting, Xu Yong. Survey of Audio-Driven Talking Face Video Generation and Identification[J]. Journal of Computer Research and Development, 2025, 62(10): 2523-2544.

[8] 宋一飞, 张炜, 陈智能, 等. 数字说话人视频生成综述[J]. 计算机辅助设计与图形学学报, 2023.

[9] 乐铮, 胡永婷, 徐勇. 音频驱动的说话人面部视频生成与鉴别综述[J]. 计算机研究与发展, 2024, 62(10): 2523-2544.

[10] Su M, Wang X, et al. Quality Assessment for Talking Head Videos via Multi-modal Feature Representation[C]//CVPR Workshop (NTIRE 2025). 2025.

[11] Liu X, et al. NTIRE 2025 XGC Quality Assessment Challenge: Methods and Results[J]. arXiv preprint arXiv:2506.02875, 2025.

[12] Quignon N. THEval: Evaluation Framework for Talking Head Video Generation[C]//OpenReview, 2024.

[13] Guo P, et al. Digital Human Techniques for Education Reform[C]//Proceedings of the 2024 7th International Conference on Educational Technology Management. ACM, 2024.

[14] 中共中央国务院. 数字中国建设整体布局规划[Z]. 2023.

[15] 教育部. Overview of work on digital education in China[R]. 2024.

[16] Chen S, Li G, Dong Y, et al. A No-reference Quality Assessment Metric for Dynamic 3D Digital Human[J]. Displays, 2023, 79: 102547.

[17] 张冰源, 张旭龙, 王健宗, 等. 数字说话人脸生成技术综述[J]. 大数据, 2024.

[18] Yu Q. Speaking Digital Person Video Generation Methods Review Report Talking Head[C]//Proceedings of the 2nd International Conference on Data Science and Engineering. 2025.

[19] SenseTime. Information Technology - General Technical Requirements for Customer Service-oriented Virtual Digital Humans (GB/T 46483-2025)[S]. 2025.

[20] Liu Q. Advancements in Digital Humans for Recorded Courses: Enhancing Learning Experiences via Personalized Interaction[J]. Frontiers of Digital Education, 2025.

[21] Goodfellow I J, Pouget-Abadie J, Mirza M, et al. Generative Adversarial Networks[C]//Advances in Neural Information Processing Systems. 2014: 2672-2680.

[22] Chung J S, Zisserman A. Out of Time: Automated Lip Sync in the Wild[C]//Asian Conference on Computer Vision. Springer, Cham, 2016: 251-263.

[23] Zhao J, Xiong X, Jayashree K, et al. Towards High Fidelity Face Frontalization in the Wild[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 2256-2265.

[24] Xie L, Zhang C, Gao Y, et al. CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 12780-12790.

[25] Wang T C, Mallya A, Liu M Y. One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 10039-10049.

[26] Guo Y, Yang C, Rao A, et al. AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 5784-5794.

[27] Blanz V, Vetter T. A Morphable Model for the Synthesis of 3D Faces[C]//Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques. 1999: 187-194.

[28] Wang X, Li Y, Zhang H, et al. Towards Real-World Blind Face Restoration with Generative Facial Prior[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 9168-9178.

[29] Li J, Zhang J, Bai X, et al. Efficient Region-aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 7534-7544.

[30] Chae H, Lee J, Nam J, et al. FaceDiffuser: Speech-Driven 3D Facial Animation Synthesis Using Diffusion[C]//ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2023: 1-5.

[31] Shen J, Yang Z, Xiang W, et al. DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 19899-19909.

[32] Fan Y, Lin Z, Saito J, et al. FaceFormer: Speech-Driven 3D Facial Animation with Transformers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 18770-18780.

[33] Kim G, Seo K, Cha S, et al. NeRFFaceSpeech: One-shot Audio-driven 3D Talking Head Synthesis via Generative Prior[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2024: 7043-7052.

[34] Yu J, Li X, Chen J, et al. GPT-SoVITS: A Versatile Voice Cloning Framework with Few-Shot Learning[J]. arXiv preprint arXiv:2402.00759, 2024.

[35] Zhang Y, Pan S, He Y, et al. MuseTalk: Real-time High-Fidelity Video Dubbing via Spatio-Temporal Sampling[J]. arXiv preprint arXiv:2410.10122, 2025.

[36] Wang Z, Bovik A C, Sheikh H R, et al. Image Quality Assessment: From Error Visibility to Structural Similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.

[37] Heusel M, Ramsauer H, Unterthiner T, et al. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium[C]//Advances in Neural Information Processing Systems. 2017: 6626-6637.

[38] Chen L, Maddox R K, Duan Z, et al. Hierarchical Cross-modal Talking Face Generation Using Latent Space Translation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 7832-7841.

[39] Xu M, Li S, Cao L, et al. Assessing Visual Quality of 3D Talking Head via Eye-tracking[C]//2023 IEEE International Conference on Multimedia and Expo. IEEE, 2023: 165-170.

[40] Ramachandram D, Taylor G W. Deep Multimodal Learning: A Survey on Recent Advances and Trends[J]. IEEE Signal Processing Magazine, 2017, 34(6): 96-108.

[41] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

[42] Wang X, Wang H, Ni H, et al. MutualFormer: Multi-Modality Representation Learning via Cross-Diffusion Attention[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 5117-5126.

[43] Liu F, Fu Z, Wang Y, et al. TACFN: Transformer-based Adaptive Cross-modal Fusion Network for Multimodal Emotion Recognition[J]. arXiv preprint arXiv:2505.06536, 2025.

[44] Jolliffe I T, Cadima J. Principal Component Analysis: A Review and Recent Developments[J]. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 2016, 374(2065): 20150202.

[45] Zhang Y, Jiang H, Miura Y, et al. Contrastive Learning of Medical Visual Representations from Paired Images and Text[C]//Machine Learning for Healthcare Conference. PMLR, 2022: 2-25.

[46] Caruana R. Multitask Learning[J]. Machine Learning, 1997, 28(1): 41-75.

[47] Misra I, Shrivastava A, Gupta A, et al. Cross-stitch Networks for Multi-task Learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 3994-4003.

[48] Chen Z, Badrinarayanan V, Lee C Y, et al. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks[C]//International Conference on Machine Learning. PMLR, 2018: 794-803.

[49] Kendall A, Gal Y, Cipolla R. Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 7482-7491.

[50] Yu T, Kumar S, Gupta A, et al. Gradient Surgery for Multi-task Learning[C]//Advances in Neural Information Processing Systems. 2020: 5824-5836.

[51] Sener O, Koltun V. Multi-Task Learning as Multi-Objective Optimization[C]//Advances in Neural Information Processing Systems. 2018: 525-536.

[52] Shen J, Pang R, Weiss R J, et al. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions[C]//2018 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2018: 4779-4783.

[53] Kim J, Kong J, Son J. Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech[C]//International Conference on Machine Learning. PMLR, 2021: 5530-5540.

[54] Jia Y, Zhang Y, Weiss R, et al. Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis[C]//Advances in Neural Information Processing Systems. 2018: 4485-4494.

[55] Zeghidour N, Luebs A, Omran A, et al. SoundStream: An End-to-End Neural Audio Codec[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021, 30: 495-507.

[56] Li Y, Han X, Zheng Z, et al. StyleTalk: One-shot Talking Head Generation with Controllable Speaking Styles[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(2): 1357-1365.

[57] Zhou Y, Han X, Shechtman E, et al. MakeItTalk: Speaker-Aware Talking-Head Animation[J]. ACM Transactions on Graphics, 2020, 39(6): 1-15.

[58] Fielding R T. Architectural Styles and the Design of Network-based Software Architectures[D]. University of California, Irvine, 2000.

[59] Nie X, Wang X, Gao Y, et al. A Review on B/S Architecture Software Development Technology[J]. Journal of Physics: Conference Series. IOP Publishing, 2021, 1952(4): 042028.

[60] Kluyver T, Ragan-Kelley B, Pérez F, et al. Jupyter Notebooks-a Publishing Format for Reproducible Computational Workflows[C]//Positioning and Power in Academic Publishing: Players, Agents and Agendas. 2016: 87-90.

[61] Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network[J]. arXiv preprint arXiv:1503.02531, 2015.

[62] Vanholder B. Efficient Inference with TensorRT[J]. NVIDIA White Paper, 2016, 1: 2.

[63] Bellard F. FFmpeg Multimedia Framework[J]. URL: http://ffmpeg.org, 2012.

[64] Pérez P, Gangnet M, Blake A. Poisson Image Editing[J]. ACM Transactions on Graphics, 2003, 22(3): 313-318.

[65] Bergkvist A, Burnett D C, Jennings C, et al. WebRTC 1.0: Real-time Communication Between Browsers[J]. World Wide Web Consortium (W3C), 2018.