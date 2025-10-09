# 心梗数据集整理

## Screening and diagnosis of cardiovascular disease using artificical intelligence-enabled cardiac magnetic resonance imaging

CMR(心脏磁共振影像) : 单通道灰度图像

1. cine MRI: 一段20到30帧的包含一个心动周期的视频
  1. SAX cine（短轴序列，9 个切片 × 25 帧）
  2. 4CH cine（长轴序列，25 帧）
2. LGE　MRI: 一组8-15张的灰度图像，合起来可以三维重建一个心脏
  1. SAX LGE（短轴，8-15张图像） 

二分类问题：判断心脏是否异常

多分类问题：输出具体的11种CVD

处理模型：使用三分支VST

## Detecting structural heart disease from electrocardiograms using AI

[echo next 数据集](https://physionet.org/content/echonext/1.1.0/)

124万对ECG-Echo数据,包括如下信息

1. 12路ECG电信号图
2. ECG测量设备生成的一些特征：心房率（atrial rate）
 - 心室率（ventricular rate）
 - PR 间期
 - QRS 宽度
 - QTc（校正 QT 间期）
3. 年龄性别

多分类问题:判断获得各种疾病的概率
1. 患SHD概率
2. 患有SHD的种类概率:
  - LVEF ≤ 45%
  - 左室壁厚 ≥ 1.3 cm
  - 中/重度右室功能不全
  - 肺动脉高压（PASP ≥ 45 mmHg 或 TR Vmax ≥ 3.2 m/s）
  - 中/重度瓣膜病（主动脉、二尖瓣、三尖瓣、肺动脉瓣）
  - 中/大量心包积液

基于CNN

## Artificial intelligence applied to electrocardiogram to rule out acute myocardial infarction: the ROMIAE multicentre study

ECG波形数据，12路波形

输出：AMI概率，二分类任务

基于ResNet

## Multimodal AI to forecast arrhythmic death in hypertrophic cardiomyopathy

MAARS:
  - EHR 分支，病人电子病历信息,表格类型实数，使用FFN层处理
  - CIR 分支，心脏影像学报告,表格类型实数，FFN层处理
  - LGE-CMR分支，三维MRI影像,使用3D ViT进行处理

问题输出依然是一个分类问题，即输出一个患有SCDA的风险概率

## Artificial intelligence-based myocardial infarction diagnosis: a comprehensive review of modern techniques

1. ECG
  - PTB dataset , 500
  - MIT-BIH Arrhythmia , 48
  - European ST-T & Long-Term ST , 87
  - Fantasia , 40
  - St. Petersburg , 75
2. 核磁共振影像
  - ACDC Dataset , 100 , cine-MRI , 心腔心肌分割，功能评估
  - EMIDEC , 150 , MI区域分割
  - Cardiac Perfusion Dataset , 35
3. 超声心动图 Echo
  - HMC-QU dataset , 二维灰度视频 , 左心室壁分割，MI检测
4. 病理切片
  - Histopathology Dataset ， 数百张切片 ， 做正常急性陈旧类型的MI分类
5. 临床数据
  - UMMC Clinical Dataset ， 140名AMI患者 ， 风险预测 临床辅助诊断
6. 生化指标
7. 非结构化电子病历数据
  

