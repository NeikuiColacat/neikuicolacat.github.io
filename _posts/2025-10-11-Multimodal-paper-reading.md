# 📚 精简阅读清单（仅限 Qibin Hou 的多模态工作）

## Cascade-CLIP (ICML 2024) → 研究图像-文本对齐，适合理解 embedding 层的脆弱性

利用类似Unet的思路 ， 使用多个level的特征来做语义分割

一层一层transformer block 先进入一个NGA，然后分别接入image-text decoder，然后再把多级特征加起来做一个softmax

复习一下focal loss 和 dice loss

dice loss ： 衡量两个集合的相似度，在分割任务里即预测掩码和真实掩码重叠程度

$$
\text{DiceLoss} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}
$$

focal loss ：用于正负样本不平衡的loss

$$
\text{FocalLoss}(p_t) = - \alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

使用vsual prompt tuning技术微调模型，在每个视觉block输入插入一些可更新参数的prompt tokens，用于微调CLIP

将若干个transformer block看作一个特征阶段，同一个特征阶段内使用NGA进行特征融合 ， 实现多层特征加权融合

## Unbiased Region-Language Alignment (ICCV 2025) → 针对开放词汇密集预测的 region-word 对齐偏差，适合研究区域级攻击。

VLM在密集预测任务上，容易把背景当成前景

DenseVLM 提出基于局部区域-文本对训练 , 使用PLM + ULM结合

PLM是一个预训练的强大VLM，对图像局部区域打伪标签

之后使用自蒸馏机制，PLM是教师模型，ULM是学生模型，训练ULM学习PLM的伪标签

问题：VLM 容易把背景区域当作前景（前景偏置）。

解决：在损失函数中显式区分前景与背景：

背景抑制项：低置信度或无法可靠匹配到类别的区域则被视为背景，进入背景抑制正则项 。  对背景区域 , 要求其与所有类别文本嵌入保持低相似度：

最后的loss ： ITC loss + 背景抑制项目 + 自蒸馏一致性损失 KL散度

## Re-Aligning Language to Visual Objects (ICLR 2025) → 引入 agentic workflow 的视觉-语言对齐，适合探索 LLM 驱动的多模态攻击。

存在问题 ： 语言驱动目标检测（Language-based Object Detection, LOD）。它指出现有方法依赖 VLM 自动生成对象描述，但这些描述常常出现“幻觉”（错误类别、颜色、形状），导致语言与视觉对象对齐不准。

解决方案 ：

1. 使用VLM生成自然语言描述
2. 使用代理模块检查语言描述是否与图像一致
3. 发现错误则自动修改描述，直到通过一致性验证

构造了一个real-LOD数据集，在类似GLIP等LOD模型去做测试

LOD模型设计架构 ：

1. 做一个视觉编码 ， 一个文本编码
2. 编码完成之后丢到一个transformer里做cross att
3. 卷出来的特征patch再丢给下游的目标检测器算出检测框

## DFormer (ICLR 2024)

![picture 0](/images/25559cd71df42575c4b8d54688d2a1bf64c877ff789af637c1f37b788f059eeb.png)  

1. RGB-D 预训练 , 在 ImageNet-1K 上生成 RGB-Depth 对（利用深度估计器 Adabins）。backbone 从一开始就学会 RGB 与深度的联合表征。

2. RGB-D Block 在编码器内部引入 RGB 与深度的交互模块，而不是在外部额外融合。

包含两个关键模块：

GAA (Global Awareness Attention)：融合 RGB 与深度的全局关系，增强 3D 感知。

LEA (Local Enhancement Attention)：用大卷积核在深度特征上提取局部几何细节，再调制 RGB 特征。

![picture 1](/images/7f09604b8229586a952dd0b904af7b3e8212491a4408e4bb2e184ca711742e80.png)  

---

跨模态一致性： “在多模态模型里，单模态扰动往往会被另一模态纠偏。如果我只在 RGB 上做攻击，如何设计损失函数才能最大化破坏跨模态一致性？”

通用攻击： “是否存在一种只基于 RGB 的扰动，但能在不同多模态任务上都迁移有效的攻击方式？”

融合层脆弱性： “像 DFormer 的 GAA/LEA 或 LOD 的 cross-attention，这些融合层是不是攻击的最佳切入点？有没有已有工作专门分析过融合层的鲁棒性？”

“如果我要证明‘单模态攻击无效，但多模态协同攻击有效’，在实验上应该如何设计对照组，才能让结论更有说服力？”

<!-- ### Sm3det (arXiv 2024) → 遥感多模态检测（RGB+SAR+高光谱） -->

<!-- MedSeg-R (2025) → 医学影像多模态分割（CT/MRI+文本/多通道）

Docopilot (CVPR 2025) → 文档级多模态理解（图像+文本+布局）

TempSamp-R1 (NeurIPS 2025) → 视频 LLM 时序采样

LLaVA-Scissor (2025) → 视频 LLM token 压缩 -->
