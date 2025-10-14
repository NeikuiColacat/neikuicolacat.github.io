# 📚 精简阅读清单（仅限 Qibin Hou 的多模态工作）

## 1. 视觉-语言对齐（基础机制，必读）

### Cascade-CLIP (ICML 2024) → 研究图像-文本对齐，适合理解 embedding 层的脆弱性

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

Unbiased Region-Language Alignment (ICCV 2025) → 针对开放词汇密集预测的 region-word 对齐偏差，适合研究区域级攻击。

Re-Aligning Language to Visual Objects (ICLR 2025) → 引入 agentic workflow 的视觉-语言对齐，适合探索 LLM 驱动的多模态攻击。

👉 价值：这三篇帮你掌握“跨模态对齐”的核心机制，是通用攻击的理论基石。
2. RGB-D / 多视觉模态融合（模态不一致攻击）
DFormer (ICLR 2024)

DFormerV2 (CVPR 2025)

👉 价值：RGB + Depth 融合，适合研究 模态不一致攻击（只攻击深度或 RGB）。这类攻击思路可以迁移到医学影像、遥感等多模态场景。
3. 应用场景（验证攻击的普适性）
Sm3det (arXiv 2024) → 遥感多模态检测（RGB+SAR+高光谱）

MedSeg-R (2025) → 医学影像多模态分割（CT/MRI+文本/多通道）

Docopilot (CVPR 2025) → 文档级多模态理解（图像+文本+布局）

👉 价值：这三篇覆盖 遥感 / 医学 / 文档 三个高风险应用场景，你的攻击方法如果能在这些任务上都有效，就能证明“跨任务通用性”。
4. 视频多模态（前沿探索，可选）
TempSamp-R1 (NeurIPS 2025) → 视频 LLM 时序采样

LLaVA-Scissor (2025) → 视频 LLM token 压缩

👉 价值：如果你想把攻击扩展到视频 LLM，可以读这两篇；但如果时间有限，可以放在最后。

---

✅ 最终精简路线（7–8 篇核心论文）
Cascade-CLIP

Unbiased Region-Language Alignment

Re-Aligning Language to Visual Objects

DFormer / DFormerV2

Sm3det

MedSeg-R

Docopilot （可选：TempSamp-R1, LLaVA-Scissor