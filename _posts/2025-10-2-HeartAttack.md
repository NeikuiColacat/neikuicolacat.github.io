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

