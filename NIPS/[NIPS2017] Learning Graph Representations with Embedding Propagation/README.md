# Learning Graph Representations with Embedding Propagation

------

## Motivation

​	在本文中，提出了一种针对图数据的，无监督学习方法——Embedding Propagation（EP）这种方法通过在图中邻居节点之间的两种信息传递来完成节点表示的学习，前向信息由节点的标签信息组成，其用来重建当前节点的表示，而后向信息则由梯度组成，是重建过程的损失。经过这样简单的信息传递就可以在更少的参数和超参数的前提下，得到比其他无监督或者半监督图嵌入方法更好的结果。

## Model

​	在介绍模型之前，首先需要明确一个概念：节点的标签信息。在本文中节点的标签信息与节点的类别标签不同，后者通常是模型需要预测的目标，而前者在本文中表示节点所具有的原始特征的集合，如下图中的一个例子，在一个学术引用网络中，节点v和其周围的邻居所具有的标签信息如图。其中每一个节点有两种类型的标签，一种标识其唯一的文章编号，另一种则为文章中出现的关键词。

![image-20200328150952447](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328150952447.png?raw=true)

​	我们的模型就是将图结构以及每一个节点所伴随的多种标签信息一起，通过无监督的方式得到一个节点的表示。完整的EP模型需要两步：

​	第一，将每一种类别的信息沿着边传递给其邻居，从信息交换过程中得到类别的表示。

​	第二，将一个节点的各个类别表示进行聚合得到最终的节点表示。

在第一步中，假设图中节点供有k个标签类别，则对于当前节点v来说，有两种得到标签表示的方式，即使用自身的标签和使用邻居的表示：
$$
 h_i (v) = g_i(\{l|l\in l_i(v)\}) \\
 \tilde h_i (v) = \tilde g_i(\{l|l\in l_i(N(v))\})
$$
其中li（v）代表节点v的第i类标签，将其对应的嵌入一并输入gi函数之后得到i类别的类别嵌入，第二个式子则是使用所有的邻居节点的同类标签来生成当前的标签表示。g函数可以是一个神经网络或者是一个Pooling函数，它将多个向量输入转化为一个向量的输出。基于一个图上邻域相似性假设，我们可以让上面两个表示尽可能相近来得到我们的无监督目标

![image-20200328153953445](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328153953445.png?raw=true)

一个使用上面例子可视化信息传递的过程如下图

![image-20200328154405959](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328154405959.png?raw=true)

对于一些节点的某一个类别不存在标签或者缺失的情况，可以为其赋予一个假的id作为标签，同样可以使用上面的方法来计算。

对于单词类的标签，可是将其表示初始化为词嵌入。而对于图片标签，可以使用一个预训练的CNN来输出其嵌入。对于随机初始化的标签嵌入，依然可以得到一个不错的训练结果。

得到了节点标签的表示之后，第二步便是聚合得到节点的表示，使用一个池化函数即可。

![image-20200328155328187](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328155328187.png?raw=true)

基于上面这样的思想，还有一种进行变形简化之后的EP模型，我们称其为Ep-B。其将信息传递过程简化为求平均值，将重建损失改为了margin Loss

![image-20200328155514206](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328155514206.png?raw=true)

其核心思想就是让当前节点的标签表示和邻居重建的标签表示的距离小，而重建表示与其他节点的表示距离大，最终将各个不同域即不同标签的嵌入连接即可得到节点的嵌入

![image-20200328155934633](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200328155934633.png?raw=true)

## Experiments

### Datasets

**BlogCatalog** 一个博客社交关系数据集 类别标签为用户的兴趣

**PPI** 蛋白质网络数据集

**POS** 词共现关系数据集

**Cora** **Citeseer** **Pubmed** 引用关系数据集

### Baselines

- **DeepWalk**
- **LINE**
- **NODE2VEC**
- **PLANETOID**
- **GCN**
- **wvRN** 用邻居标签的平均值作为本节点标签，这里用邻居中出现最多的标签
- **MAJORITY** 总是输出训练集中出现最多的标签

### Results

![image-20200329105315745](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Learning%20Graph%20Representations%20with%20Embedding%20Propagation/image-20200329105315745.png?raw=true)

## 个人见解

​	本文提出的模型叫做嵌入传播，实际上就是一个无监督改进版的GCN，GCN中聚合邻居节点的信息来与最终的标签交互产生Loss，而无监督版本就是把这个标签改为了节点自己的信息，这种重建损失作为无监督损失是一种常见的思路，这一点可以作为今后需要在图上的无监督模块时的一个可选项，跟DeepWalk的思想是一脉相承的。有一点不可思议的是，这种无监督方法竟然比有监督额GCN还要好，如果实验结果没问题的话，确实是想不到的。