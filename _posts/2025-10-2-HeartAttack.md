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
PTB dataset , 500
MIT-BIH Arrhythmia , 48
European ST-T & Long-Term ST , 87
Fantasia , 40
St. Petersburg , 75
2. 核磁共振影像
ACDC Dataset , 100 , cine-MRI , 心腔心肌分割，功能评估
EMIDEC , 150 , MI区域分割
Cardiac Perfusion Dataset , 35
3. 超声心动图 Echo
HMC-QU dataset , 二维灰度视频 , 左心室壁分割，MI检测
4. 病理切片
Histopathology Dataset ， 数百张切片 ， 做正常急性陈旧类型的MI分类
5. 临床数据
UMMC Clinical Dataset ， 140名AMI患者 ， 风险预测 临床辅助诊断

---

相关数据集

1. MyoPS 2020 ， 图像分割， 分类区分正常梗死，区分急性慢性，预后检测

45 位患者的多序列心脏 MRI（T2、bSSFP、LGE）影像

人工标注的 心肌瘢痕、水肿、正常心肌、左右心室血池 图像分割任务
2. EMIDEC , 图像分割， 分类区分正常梗死

150 例延迟增强心脏 MRI（DE-MRI）病例，其中 100 例为心肌梗死患者，50 例为正常对照

人工分割的解剖/病灶区域包括：

左心室 (LV)

心肌 (Myocardium)

心肌梗死 (Myocardial infarction)

无复流区 (No-reflow, microvascular obstruction)
3. SCMR Consensus，心肌轮廓分割，
4. ACDC (Automated Cardiac Diagnosis Challenge)，图像分割

100训练集 带有标注 ， 50测试集无标注 ， 带有心梗疾病5个类型，心脏各类性能指标参数

正常 (Normal)

心力衰竭伴梗死 (Heart failure with infarction)

扩张型心肌病 (Dilated cardiomyopathy)

肥厚型心肌病 (Hypertrophic cardiomyopathy)

右心室异常 (Right ventricular abnormality)
5. M&M(Multi-Centre, Multi-Vendor) ， 心肌分割标注

150公开 训练集，225测试集(无标签信息)

NOR：正常（Normal）

DCM：扩张型心肌病（Dilated Cardiomyopathy）

HCM：肥厚型心肌病（Hypertrophic Cardiomyopathy）

MINF：心肌梗死（Myocardial Infarction）

RV：右心室异常（Right Ventricular Abnormality）
6. CMRxRecon 2024 , MRI扫描仪的原始k空间数据,330个样本，用于AI快速心脏成像
7. Cardiac Atlas Project (CAP) , MRI图像数据 ， 解剖结构分割，功能参数，病理分类，临床患者信息
8. PTB-XL
9. EchoNet-Dynamic
10. Cleveland
11. CAMUS

---

## 具体医学任务

### 诊断类型任务

1. MI检测(二分类任务)

主要依靠12导联ECG

LGE MRI

电子病历
2. MI多分类任务

STEMI(冠状动脉完全阻塞)

NSTEMI(不完全阻塞)

MINOCA(心肌梗死，但不完全阻塞)

ECG

CAG

LGE MRI、T2 MRI

Echo

## 病灶分析类任务

1. 病变部位识别分割分类

瘢痕：心肌完全坏死区域

灰区：心肌受损但是不完全坏死区域

远端区：正常心脏组织

壁段：前壁、下壁、侧壁、间隔部

冠脉：LAD、LCX、RCA

LGE MRI , Echo , CCTA , ECG
2. 灌注缺损标识任务(识别心肌区域是否存在血流灌注不足，提示供血障碍或梗死区域)

MRP(MRI灌注成像) 与 LGE MRI 结合
3. 风险评估类任务

短期住院期间再次梗死风险

出院后30天内再次心梗风险

1年/五年 死亡率

ECG , MRI , CTTA , Echo , 电子病历表格数据 , 可穿戴设备数据
4. 回归类型任务

LVEF , RVEF , EDV , ESV , SV , CO , CI : MRI和Echo

---

## 数据集

### 心电生理（ECG）

#### PTB-XL（大规模，2.1万名患者，12导联ECG，PhysioNet可下载 ✅ 主流）

[2020年发布，下载链接](https://physionet.org/content/ptb-xl/1.0.1/)

21,837条 ECG 记录

标准12导联

心梗类型标签，心梗病变部位标签，电子病历信息

### PTB Diagnostic ECG Database（500例，经典AMI检测数据集 ✅ 可下载）

[2004年发布，下载链接](https://physionet.org/content/ptbdb/1.0.0/)

549条记录

15导联

记录了心梗病变部位信息

### Georgia 12-lead ECG Challenge Database

[下载链接](https://www.kaggle.com/datasets/physionet/georgia-12lead-ecg-challenge-database)

10,344 条 ECG

12 导联

心肌梗死（MI）：部分记录含 MI 标签（如 STEMI、NSTEMI、前壁/下壁梗死）

### MRI

#### ACDC Dataset（100例 cine-MRI，心腔/心肌分割，功能评估 ✅ 可下载）

Cine-MRI 数据集 , 100例

[下载](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

标签提供心脏解剖结构分割信息

#### EMIDEC（150例 LGE-MRI，MI区域分割 ✅ 可下载）

[下载，需要申请](https://emidec.com/)

像素级标签包含如下内容

1. 左心室心肌区域
2. 心肌梗死区域
3. 无复流区域(灌注缺失区域)

#### MyoPS 2020（45例，多序列MRI：T2、cine-MRI、LGE，瘢痕/水肿分割 ✅ 可下载）

1. 左心室血池 LV
2. 右心室血池 RV
3. 正常心肌
4. 心肌瘢痕
5. 心肌水肿区域

#### M&M (Multi-Centre, Multi-Vendor)（150训练+225测试，心肌分割，含MINF ✅ 可下载）

[下载链接](https://www.ub.edu/mnms/)

分割标签：

1. LV（左心室腔体）

2. RV（右心室腔体）

3. Myo（左心室心肌壁）

病理分类标签：

1. NOR (正常)

2. MINF(心肌梗死)

### CT

无

### Echo

EchoNet-Dynamic（大规模，1万+视频，LVEF预测，Stanford ✅ 主流）

CAMUS（500名患者，2D Echo，心腔分割 ✅ 可下载）

HMC-QU dataset（二维灰度视频，左心室壁分割，MI检测）

EchoNext Dataset（124万对 ECG-Echo，PhysioNet ✅）

### CAG冠状动脉造影

### 电子病历

Cleveland Heart Disease Dataset（UCI库，经典小规模表格数据 ✅ 可下载）

### 可穿戴设备

Apple Heart Study、UK Biobank（需申请访问，规模大）
