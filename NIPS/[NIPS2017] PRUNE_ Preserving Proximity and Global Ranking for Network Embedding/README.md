# PRUNE_ Preserving Proximity and Global Ranking for Network Embedding

------

## Motivation

​	网络嵌入是将一个邻接矩阵代表的n×n的网络转换为一个n×d的低维特征矩阵，在之前的工作中，很多方法都是在保留节点的k阶相似度的前提下进行的节点嵌入，这样的方法能够一定程度上保留网络的局部拓扑结构，从而用户连接预测、社区发现等任务，但是除此之外，我们发现全局的节点排名信息同样是一个极为重要的指标，将其纳入嵌入中能够赋予更强的性能。所以本文中，提出一个同时保留局部相似度和全局排名的无监督网络嵌入方法，其使用非常简单的模型就可以晚上两个指标的学习。

## Model

![image-20200330151215796](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20PRUNE_%20Preserving%20Proximity%20and%20Global%20Ranking%20for%20Network%20Embedding/image-20200330151215796.png?raw=true)

​	模型的整体结构如上图所示，其中圆点代表一个标量，而方块则代表一个向量，每一个灰色的箭头头代表一个非线性的映射即一层的神经网络。

​	本模型以一条正样本边为训练实体，不需要进行负采样操作，同时模型的主体就是一个共用的隐藏层以及几个单层的神经网络，最主要的部分就是最终的目标函数，也是下面主要介绍的部分。

​	首先，将边（i，j）输入之后，分别查表得到其相应的嵌入ui和uj接下来需要得到两个特征向量，分别为图中的z和Π。得到的方式都是通过两层的神经网络。
$$
z = \phi_2 (w_2\phi_1(w_1u + b_1) + b_2)\\
\pi = \phi_4 (w_4 \phi_3(w_3u + b_3) + b_4)
$$
其中向量z用来计算节点之间的距离，而标量Π则表示节点的全局排名的分数。为了后面损失设计方便，这里我们让z和Π都是非负的，这可以通过设置特殊的激活函数来实现。

因为我们设计的是一个无监督的方法，所以损失函数定义自节点（i，j）之间。

![image-20200330152914180](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20PRUNE_%20Preserving%20Proximity%20and%20Global%20Ranking%20for%20Network%20Embedding/image-20200330152914180.png?raw=true)

其中等号前后的两项分别对应节点距离和全局排名两个任务。m代表一个节点的入读而n代表一个节点的出度，都是提前知道的整数值。

其中第一项的的含义是让两个向量的点积与他们的互信息（PMI）相互接近，这一点在本文中有证明，第二项来自于PageRank的思路，即流入和流出每一个节点的信息应当守恒，所以每一个节点的得分应当由其出边平均共享给其他节点，这样就需要让这条边上的流入等于流出。详细的证明同样在文中由柯西-西尔瓦兹不等式可以得到。

## Experiments

### Datasets

- **Hep-Ph** 这是一个论文引用网络，我们将每篇论文的引用数作为排名依据，来检测节点嵌入在全局排名上的效果
- **Webspam** 一个网站和它们之间的链接的网络，目标是将其中标为非垃圾网站的节点排名在标为垃圾网站的节点之前。
- **FB Wall Post** 一个Facebook的社交网络，需要让活跃用户排名在非活跃用户之前。

### Baselines

- DeepWalk
- LINE
- node2vec
- SDNE
- NRCL

### Results

![image-20200330161940602](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20PRUNE_%20Preserving%20Proximity%20and%20Global%20Ranking%20for%20Network%20Embedding/image-20200330161940602.png?raw=true)

## 个人见解

​	本文的工作是一个将无监督的网络嵌入与之前的PageRank相结合的工作，通过修改目标函数来统一二者不同的需求，并通过理论分析的方式找到新的目标与两个传统的老目标之间的关系，同时从互信息的角度来定义局部相似度以此来避免进行负采样。本文应该算是一个典型的NIPS的文章，非常理论，模型结构简单目标明确，但是理论分析与证明繁多，很多证明都放到了最后的补充材料中，读起来感觉特别硬。另外一个让人费解的地方是并没有在常用的数据集上面进行实验，而且全局排序的目标应用场景也不多，在其他任务中的表现不应于其有直接地联系，所以也就没有了后续工作。

