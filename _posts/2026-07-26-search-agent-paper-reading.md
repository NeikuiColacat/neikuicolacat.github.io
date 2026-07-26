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
