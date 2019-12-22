# Anonymous Walk Embeddings

------

## Introduction

​	图嵌入（Graph embedding）包含两个层次：图节点的嵌入和整个图的嵌入。这篇文章便是关注的后者。图嵌入是把整个图即G（V，E）表示为一个固定长度的向量，这样就可以把复杂的、无法直接处理的拓扑结构变为数据挖掘中易使用的数字形式，从而应用于网络结构分析和分类等任务。先前的工作使用CNN来处理网络结构数据，但这是一种监督的方法，与特定的任务相关，为了找到一种与下游的任务无关的图嵌入方法，本文提出了一种基于“匿名游走”的嵌入学习方法，可以以无监督的方式得到质量优秀的图向量。

## 匿名游走

​	匿名游走是在随机游走的基础上放弃节点的具体含义而只关注相对关系的一种图上的采样方法。与随机游走一样，这种采样的作用均是把一个图变成一些长度固定的序列（语料）。举个例子来理解这种匿名概念。

![image-20191222142520392](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222142520392.png?raw=true)

​	假设图中有A、B、C、D、E、F这些节点，每个节点与其他节点均不同，以A为起始点进行 Random Walk，可以得到Random Walk 1和Random Walk 2这两个序列，同理以2为起点得到 Random Walk 2，对于得到的每一个序列我们对其重新编号（匿名化）即如果此节点是第一次出现在序列中则为其分配一个新的id，如果之前出现过，则依旧用上次使用的id，这样对于一个节点，我们不关心它到底是A、B还是C，而只关心之前有没有出现过。先随机游走再进行匿名化，这就是匿名游走的整个采样过程。

​	一些论文证明（见原文中参考文献），这样的匿名序列是包含足够的链接信息的，即只要有足够的采样到的相同起点的匿名游走序列，那么就可以重建以起点为中心的这个子图的链接信息，这也是本文提出的用匿名游走进行图嵌入的最主要的理论依据。通过建模图上匿名游走形成的语料库，就可以完全的建模出这个图的特征。

## Model

​	有了理论依据，接下来便是设计详细的对语料进行建模的方法，但是在讲述模型之前，文章还花费了一定的篇幅解释了一个问题：匿名游走语料库怎么来？在这里只是总结一下结论，中间的证明和叙述过程参见原文。对于一个图，其不等价的匿名游走序列是与图的直径成指数关系的，而进行上述的序列生成过程是O（n）的，得到完整的语料库是要付出巨大的计算代价的，所以只能进行有限次的采样，保证采样得到的语料的分布与其真实分布误差再一定范围内。

​	有了这样的前提就可以使用固定的训练计算量来尽可能的建模图的特征了。对于得到的语料库，可以类比为NLP中的doc2vec问题，要学习的graph 相当于 doc 而一个个的序列就相当于doc中的word，这样就可以仿照Skip-gram的方式来设计具体的嵌入模型。具体的，我们把起点相同的序列定义为邻居序列，使用邻居序列和整个图来预测当前序列，具体的优化目标如下面的公式。<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222150245295.png?raw=true" alt="image-20191222150245295" style="zoom: 67%;" />
<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222150421373.png?raw=true" alt="image-20191222150421373" style="zoom:67%;" />
<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222150614796.png?raw=true" alt="image-20191222150614796" style="zoom:67%;" />

用graph vector和context vector算出当前序列的得分，在最大化softmax之后的log概率，本质上还是skip-gram的思想，只不过最终的目标是作为伴随特征的graph vector。在训练过程中同样使用负采样来减少softmax步骤的计算量。图示化的模型见下图。

<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222151209158.png?raw=true" style="zoom:80%;" />

## Experiments

​	在文章中作者除了提出使用类似Skip-gram方法对语料库进行训练的方式（data-driven）之外还提出了一种使用极大似然估计的方法来建模语料库中序列分布的方法（feature-based），分别用这两种建模方式在有标签和无标签的网络数据集中进行了实验，任务均是网络的分类问题。

### Results

![image-20191222154302337](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2018%5D%20Anonymous%20Walk%20Embeddings/image-20191222154302337.png?raw=true)

​	从结果来看，AWE相比之前的GCN和传统的核方法性能均有所提高，且使用训练embedding的data-driven方法效果更好。



## 个人见解

​	这篇文章的问题是整个图的嵌入，其任务粒度也从图中的节点变成了整个图。最让人觉得眼前一亮的地方就是将序列语料库类比到doc的过程，doc和word之间的层次关系刻画的很有创意，同时可以看到基础的Skip-gram方法的应用面还是很广的，不能被NLP所局限。最后我个人觉得一点美中不足的地方是学习到的匿名游走序列embedding被丢弃而没有使用，如何找到相比于一个graph vector数量巨大的序列embedding的含义和使用方式可能是一个未来优化的方向。