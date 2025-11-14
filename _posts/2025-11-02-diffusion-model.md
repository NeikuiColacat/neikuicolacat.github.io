# 扩散模型学习

## text encoder and noise predictor 对模型性能的影响

![picture 0](/images/b0868bc94dd3958745c1077df1f4689a03e866049cca05e2106c92d34da7e3fa.png)  

note that , 图片生成质量常用FID 与 CLIP score来量化

## 主流image generation模型常见架构

![picture 1](/images/5ac40bd1bc5e85fe2c0491d3b6351ec85bf97311f3bf39d75d2cfb700d671352.png)  

note that, Generation Model一般用diffusion model来做

## 如何将VAE和diffusion 串联理解起来

![picture 2](/images/a90d15f7c7361365730ae7d4a3de7d7c92a7b8999f01a1380ba16ecfb2089b4e.png)  

VAE的encoder输出是一个latent variable(刻画d维高斯分布参数)

Diffusion 加噪声的过程也可以理解为在做encoding ， 输出加完噪声的图片也是 latent variable

## DDPM 算法

### encoding 过程

![picture 3](/images/e6a821b2542f91804bb19f8be2e4e3b3913e2828ce069edae656d2a817383cd8.png)  

### decoding 过程

![picture 4](/images/df0c6b24fdd66d91b932e28bcac6d4ad1322633ac8a2d91da4c28ac38d3423cf.png)  

---

生成模型在做sampling的时候并不会选取几率最大的那个prediction 。 例如在GPT中如果这么做，那么就会一直重复说一样的话

![picture 5](/images/7171ac674bcc54f9ccf3ce2e45db4b85f0de2209f01eac7760e4d8d8b785afb0.png)  

同理对于diffusion model来说，我们也需要随机的sampling

![picture 6](/images/038545a0426921ba6f92ce6ea5626f77143473c6f75a66a04afb2ffb81a7bde6.png)  

![picture 7](/images/a8494b4439a200f8dd6da1d046547418451bb97c1eab3dfa2c531ebff8918202.png)  

## Latent Diffusion Models

使用latent space中采样的噪声来迭代去噪

![picture 8](/images/2e2721b8c2b39cff87cd0d045dcd9ba30f43257cbc8507d2bc5da1b90268e51a.png)  

第一个阶段先训练一个autoencoder去压缩图像到一个latent space

去噪模型用到是一个Unet + Crossattention结构做条件生成

注意到加噪和去噪训练过程都是在latent space上进行的

最终目的是想要能让模型从latent space中借助我们给定的条件去生成一个latent variable

然后通过autoencoder 的decoder部分再生成我们想要的图片

## 关于diffusion模型的引导问题

### 方法一 classifier guidance

训练一个分类器，在加噪声的基础上预测最后生成的图像类别是什么，求得我想要对应类别的梯度，在去噪的时候将噪声加上一个想要的梯度

### 方法二 classifier free (GLIDE)

除了常规带上文本训练之外，一定概率（10%）不要带上文本去训练

通过调整权重系数去平衡fidelity和diversity

classifier free方法普遍要更好，社区主流

![picture 10](/images/f3373d524c197184402588a90a3f11a458259926ca3ef75b61adb1098af77df0.png)  

引导方向 = 条件模型的梯度 - 无条件模型的梯度

## ECGTwin

![picture 9](/images/e1b8c0143aa055939b39b97eccc12d790a9d9cb9cd578bb8fdeeb4ede3593541.png)  

### 目标任务 ：已知一个病人的ECG，当前的心脏状态，生成他在另一个心脏状态下的ECG信息

### Individual Base Extractor

把患者在不同心脏状态下都一致的生理特征抽成一个base vec出来，用来个diffusion生成提供ref

构造CLIP风格的自监督对比学习，让同一个患者的不同时间或状态的base vec余弦相似度拉进

### 去噪ref路径一

每个临床报告使用nomic-embed-text-v1.5编码为一个token拼接

构造一个向量P： sex使用二值编码 ， age，heart rate使用normalization

之后将P嵌入到每个报告token前面形成增强token

之后所有增强token堆叠为一个序列，再加入位置编码

之后使用一个Linear层将堆叠的增强token序列映射到一个embedding space下，并加上位置编码，用于区分每个报告

### 去噪ref路径二

使用 DiT（Diffusion Transformer） 中的 sinusoidal encoding 方法得到一个时间步向量t

将base vec映射一下和t维度对齐，将两个向量相加得到R

R输入MLP得到 α β γ三个缩放因子，将latent 表示 z进行layer norm

## 关于SDXL的一些训练技巧

文本encoder非常关键 ，SDXL拿两个CLIP拼了一个超大的word embedding

可以参考base + refiner结构，这样输出的东西会更加精细一些

把图片分辨率加入我们的生成condition里面

把训练时候图片crop的坐标加入生成conditon里面(左上角起始像素位置)

## DiT 架构解析

![picture 11](/images/1c35e40625c0025357b758e6d9e85ddf256fe50b7d05b9bd12c9b3c4e480a179.png)  

训练时MLP首先会被置为0，辅助稳定训练
