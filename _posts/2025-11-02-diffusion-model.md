# 扩散模型学习

## text encoder and noise predictor 对模型性能的影响

![picture 0](../images/b0868bc94dd3958745c1077df1f4689a03e866049cca05e2106c92d34da7e3fa.png)  

note that , 图片生成质量常用FID 与 CLIP score来量化

## 主流image generation模型常见架构

![picture 1](../images/5ac40bd1bc5e85fe2c0491d3b6351ec85bf97311f3bf39d75d2cfb700d671352.png)  

note that, Generation Model一般用diffusion model来做

## 如何将VAE和diffusion 串联理解起来

![picture 2](../images/a90d15f7c7361365730ae7d4a3de7d7c92a7b8999f01a1380ba16ecfb2089b4e.png)  

VAE的encoder输出是一个latent variable(刻画d维高斯分布参数)

Diffusion 加噪声的过程也可以理解为在做encoding ， 输出加完噪声的图片也是 latent variable

## DDPM 算法

### encoding 过程

![picture 3](../images/e6a821b2542f91804bb19f8be2e4e3b3913e2828ce069edae656d2a817383cd8.png)  

### decoding 过程

![picture 4](../images/df0c6b24fdd66d91b932e28bcac6d4ad1322633ac8a2d91da4c28ac38d3423cf.png)  

---

生成模型在做sampling的时候并不会选取几率最大的那个prediction 。 例如在GPT中如果这么做，那么就会一直重复说一样的话

![picture 5](../images/7171ac674bcc54f9ccf3ce2e45db4b85f0de2209f01eac7760e4d8d8b785afb0.png)  

同理对于diffusion model来说，我们也需要随机的sampling

![picture 6](../images/038545a0426921ba6f92ce6ea5626f77143473c6f75a66a04afb2ffb81a7bde6.png)  

![picture 7](../images/a8494b4439a200f8dd6da1d046547418451bb97c1eab3dfa2c531ebff8918202.png)  
