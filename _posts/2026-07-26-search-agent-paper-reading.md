---
layout: post
title: Search Agent 论文阅读记录
date: 2026-07-26
description: Paper reading notes on search agents, including query planning, retrieval, reflection, and evaluation
tags: [ml, agent]
giscus_comments: false
---

## 阅读列表

- [阅读列表](#阅读列表)
- [FlowSearch: Advancing Deep Research with Dynamic Structured Knowledge Flow](#flowsearch-advancing-deep-research-with-dynamic-structured-knowledge-flow)
  - [基本信息](#基本信息)
  - [研究背景与问题](#研究背景与问题)
  - [方法总览](#方法总览)
  - [关键图表记录](#关键图表记录)
  - [我的理解与思考](#我的理解与思考)
- [Actions Speak Louder Than Prompts: A Large-Scale Study of LLMs for Graph Inference](#actions-speak-louder-than-prompts-a-large-scale-study-of-llms-for-graph-inference)
  - [基本信息](#基本信息-1)
  - [研究背景与问题](#研究背景与问题-1)
  - [方法总览](#方法总览-1)
  - [关键图表记录](#关键图表记录-1)
  - [我的理解与思考](#我的理解与思考-1)
- [PPO 算法](#ppo-算法)
  - [目标 ： 希望最大化期望 reward](#目标--希望最大化期望-reward)
  - [存在问题](#存在问题)
  - [importance sampling](#importance-sampling)

## FlowSearch: Advancing Deep Research with Dynamic Structured Knowledge Flow

### 基本信息

- 阅读日期：2026-07-26
- 作者：Yusong Hu、Runmin Ma、Yue Fan、Jinxin Shi、Zongsheng Cao、Yuhao Zhou、Jiakang Yuan、Shuaiyu Zhang、Shiyang Feng、Xiangchao Yan、Shufei Zhang、Wenlong Zhang、Lei Bai、Bo Zhang
- 机构：上海人工智能实验室（Shanghai Artificial Intelligence Laboratory）
- 会议 / 年份：ACL 2026，Long Papers
- 论文链接：[ACL Anthology](https://aclanthology.org/2026.acl-long.971/) · [arXiv](https://arxiv.org/abs/2510.08521)
- DOI：[10.18653/v1/2026.acl-long.971](https://doi.org/10.18653/v1/2026.acl-long.971)
- 项目 / 代码：[InternScience/InternAgent](https://github.com/InternScience/InternAgent)

### 研究背景与问题

agent 线性规划 提出一个 长链结构的graph (很多个假设节点) 然后按照顺序去做 ， 但是做的过程中这个 plan graph 会不断变动的

所以提出 动态构建 graph DAG

### 方法总览

做三个 agent ： planner agent 负责构建 plan graph ， collector agent 负责执行子节点 ， refiner 负责 collector执行完成之后 动态修改graph

1. planner 将用户的问题转化为一个初始的 plan graph ， 拆成多个待执行的节点
2. collector 负责执行 plan graph 中的节点，收集证据和信息 ， 然后做完的节点标记为完成
3. refiner 根据 collector 收集到的证据和信息，动态修改 plan graph

### 关键图表记录

![FlowSearch 方法总览](/assets/img/1785071297600.png)

### 我的理解与思考

验证了我的想法 ， 让agent 一边 工作 一边 动态构建 graph ， 让grpah 指导agent做事情 ， 将graph作为一个短期记忆是对的

但是我想干的是让 agent 自行构建一个搜索时的 知识图谱 ， 知识图谱 能给 agent hint 对当前搜索进度和整体大局有一个view吗 ？ graph 对 agent 来说感觉不如人类看起来直观怎么办 ， 人类看grpah直观是因为graph是视觉化 ， 给agent喂graph只能通过文本描述这个graph ？

对的对的 ， 思考一下 在金融领域应用 ， graph这个东西应该如何 喂给 agent 是最好的

## Actions Speak Louder Than Prompts: A Large-Scale Study of LLMs for Graph Inference

### 基本信息

- 阅读日期：2026-07-27
- 作者：Ben Finkelshtein、Silviu Cucerzan、Sujay Kumar Jauhar、Ryen W. White
- 机构：牛津大学（University of Oxford）、微软研究院（Microsoft Research）
- 会议 / 年份：ICLR 2026，Oral
- 论文链接：[OpenReview](https://openreview.net/forum?id=MgJUj9Sk3C) · [arXiv](https://arxiv.org/abs/2509.18487) · [ICLR](https://iclr.cc/virtual/2026/poster/10009927)
- DOI：暂无
- 项目 / 代码：[Microsoft Research 论文介绍](https://www.microsoft.com/en-us/research/publication/actions-speak-louder-than-prompts-a-large-scale-study-of-llms-for-graph-inference/) · 代码暂未公开

### 研究背景与问题

对于graph结构的数据来说应该给agent提供什么样的交互接口是好的

takeaway : 让agent自己写代码操纵graph的效果是最好的

### 方法总览

实验了一下三种方法 ：

1. 直接把graph用提示词的形式喂给agent
2. 给agent提供操纵graph的tool接口
3. 直接让agent自行写代码操纵graph

### 关键图表记录

![Prompting、GraphTool 与 Graph-as-Code 三种交互方式](/assets/img/1785144011347.png)

### 我的理解与思考

原作者是在做图节点分类任务的 ， 可能迁移到agent搜索领域风险较大 ， 但是让agent写代码操纵graph的思想可以takeaway一下

当graph规模增长到一定程度的时候 ， 使用 1-2hop 将邻近节点信息塞到 agent 的上下文中不管是性能还是token消耗 都不如 直接让agent 写代码操纵graph了

## PPO 算法

### 目标 ： 希望最大化期望 reward

$$

J(\theta) =  \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]

$$

### 存在问题

策略更新依靠旧策略采样到的数据进行优化 ， 但是新策略采样到的数据分布和老策略的不一样，训练不稳定

So how to provide old data for new policy ?

introducing : importance sampling

### importance sampling

先求出新 policy 做出这个动作相比老policy 的比例关系

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

现在得到了因子 $$ r_t $$ 用于对旧采样数据进行加权 ， 这是第一个因子

同时还需要评估这个动作到底好不好 ， 也就是 $A_t$ ，用于描述相对于之前的策略当前的动作能够更好还是更坏， 这是第二个因子

$$
A_t = Q^{\pi_{\theta_{\mathrm{old}}}}(s_t, a_t) - V^{\pi_{\theta_{\mathrm{old}}}}(s_t)
$$

V 表示按照当前status能拿到的reward

Q 表示按当前status执行action之后能拿到的reward

---

So how do we get the Q ?

不知道 Q ， 那么就执行一次 action 进入到下一个状态 ， 然后就能拿到 $V (s_{t+1})$

所以 $Q(s_t , a_t) \approx r_t + \gamma V(s_{t+1}) $

继续 将 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

如果不只是往前看一步 ， 而是看多步 ， 就有

$A_t^{GAE}=\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$

$\lambda$ 用于控制 Advantage 看多远

---

两个因子相乘 得到因子，同时为了限制新策略和旧策略的差距，使用了一个clip函数来限制 $ r_t $ 范围

最终得到PPO核心目标 ：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[\min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right)\right]
$$

---

$V(s_t)$ 用来估值的也是一个神经网络 ， 需要训练

Vt loss 通常写为 

$$
L_V = (V_\theta(s_t) - R_t)^2
$$

$$
R_t = V_{old}(s_t) + A_t
$$

---

entropy bonus 用于鼓励策略探索随机性

$$
H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)
$$

### 强化学习 bg review 

首先我们有一个总的 reward 函数 ， 公式表达为 ： $J(\theta) = \mathbb{E}_{\tau \sim P_\theta} [R(\tau)] = \sum_\tau P_\theta(\tau)R(\tau)$

求导：

$$
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P_\theta(\tau)R(\tau)
$$

然后因为采样轨迹已经固定了 ， 我们对策略 $\theta$ 求导与 reward 函数无关 ， 所以我们可以把 reward 函数提出来 ， 然后对策略求导

$$
\nabla_\theta J(\theta)
= \sum_\tau R(\tau)\nabla_\theta P_\theta(\tau)
$$

接下来给 policy 的轨迹概率 $P_\theta(\tau)$ 套一层 log。因为 $\log x$ 的导数是 $1/x$，所以求导后会得到：

$$
\nabla_\theta \log P_\theta(\tau)
= \frac{1}{P_\theta(\tau)}\nabla_\theta P_\theta(\tau)
$$

将等式两边同时乘以 $P_\theta(\tau)$，policy 概率的梯度就可以等价写成：

$$
\nabla_\theta P_\theta(\tau)
= P_\theta(\tau)\nabla_\theta \log P_\theta(\tau)
$$

将这个等价形式代回目标函数的梯度：

$$
\nabla_\theta J(\theta) = \sum_\tau P_\theta(\tau)R(\tau)\nabla_\theta \log P_\theta(\tau)
$$

因为对所有轨迹按照 $P_\theta(\tau)$ 加权求和就是求期望，所以可以写成：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta}\left[R(\tau)\nabla_\theta \log P_\theta(\tau)\right]
$$

一条轨迹由多个时间步上的 action 组成。只保留与 $\theta$ 有关的 policy 部分，轨迹概率可以写成连乘形式：

$$
P_\theta(\tau) = \prod_{t=0}^{T-1}\pi_\theta(a_t \mid s_t)
$$

#### 为什么要给 policy 套一层 log？

套 log 不是为了把所有 action 都采样出来，实际训练仍然只会采样有限数量的轨迹。它的核心作用是配合前面的对数导数技巧，把 $\nabla_\theta P_\theta(\tau)$ 改写成 $P_\theta(\tau)\nabla_\theta\log P_\theta(\tau)$。这样 $P_\theta(\tau)$ 就可以作为采样分布被吸收到期望中，训练时只需要对采样得到的轨迹求平均，不需要枚举所有可能的轨迹。

另外，一条轨迹的概率是多个 action probability 的连乘。套 log 后，连乘会变成逐时间步的连加，求梯度时也就能拆成每个 action 的 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。代码中直接累加 log probability 还可以避免很多小概率连续相乘造成的数值下溢。

因此，轨迹概率套 log 后可以写成：

$$
\log P_\theta(\tau) = \sum_{t=0}^{T-1}\log \pi_\theta(a_t \mid s_t)
$$

然后对 $\theta$ 求导：

$$
\nabla_\theta \log P_\theta(\tau) = \sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

最后代回目标函数，就得到策略梯度的基本形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta}\left[R(\tau)\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
$$

实际训练时，轨迹已经按照 $P_\theta(\tau)$ 采样出来了。$P_\theta(\tau)$ 已经体现在“哪些轨迹会被采到以及被采到多少次”上，所以用采样数据估计期望时，不需要在 loss 中再次显式乘以 $P_\theta(\tau)$。

将一个 batch 中的采样数据展开成 $B$ 个 state-action 样本后，代码实际估计的策略梯度是：

$$
\widehat{\nabla_\theta J(\theta)} = \frac{1}{B}\sum_{b=1}^{B}\hat{A}_b\nabla_\theta \log \pi_\theta(a_b \mid s_b)
$$

代码中通常通过最小化负的 policy loss 来实现梯度上升，其中 advantage 不参与 policy 的反向传播：

$$
L_{\mathrm{policy}}(\theta) = -\frac{1}{B}\sum_{b=1}^{B}\operatorname{stopgrad}(\hat{A}_b)\log \pi_\theta(a_b \mid s_b)
$$

上面是数据由当前策略 $\pi_\theta$ 采样时的 on-policy 形式。PPO 实际使用旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采集的数据，因此不乘完整的轨迹概率，而是使用新旧策略的概率比进行修正(代码实现新旧policy比例需要做除法 ， 用log函数减法替代)：

$$
r_b(\theta) = \exp\left(\log \pi_\theta(a_b \mid s_b)-\log \pi_{\theta_{\mathrm{old}}}(a_b \mid s_b)\right)
$$

因此 PPO 在代码中实际最小化的 policy loss 是，其中 $\hat{A}_b$ 同样作为已经计算好的固定量，不对它进行 policy 方向的反向传播：

$$
L_{\mathrm{PPO}}(\theta) = -\frac{1}{B}\sum_{b=1}^{B}\min\left(r_b(\theta)\hat{A}_b,\operatorname{clip}(r_b(\theta),1-\epsilon,1+\epsilon)\hat{A}_b\right)
$$
