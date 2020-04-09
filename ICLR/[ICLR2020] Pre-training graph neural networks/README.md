# Pre-training graph neural networks

------

## Motivation

​	预训练模型在图像和文字领域都已经有了非常大的发展，但是在面向图数据的GNN上面还没有一个可行的预训练策略，本文正是提出了一种针对GNN的预训练方法，同时考虑节点级别和图级别的任务以使得预训练的表示能尽可能地携带两个不同级别的信息，同时为了进行大规模的预训练，我们构建了两个超大规模的图数据集分别来自化学和生物领域，包含百万量级的图信息。

## Model

​	在预训练中需要同时考虑节点级别的属性和整图级别的属性，所以下面我们分开介绍这两方便的预训练策略。

### NODE-LEVEL PRE-TRAINING

​	对于节点级别的预训练，我们采取一种自监督的方式，训练目标是让具有相似上下文结构的节点出现在相似的embedding上，首先需要定义任务，因为是无监督，所以节点不能用标签来训练嵌入，我们考虑一个节点的上下文图，其定义为距离节点v大于r1但是小于r2的那些节点集合。节点v的K跳邻居即以v为中心的距离小于K的子图

![image-20200408140250027](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2020%5D%20Pre-training%20graph%20neural%20networks/image-20200408140250027.png?raw=true)

​		与NLP预训练的思想一样，我们要用节点v的邻居信息来预测其上下文信息，对于节点v其K层的GNN便可以得到K跳邻居的表示，那么怎么得到上下文信息呢？这里我们定义锚点节点，其为那些邻居集合和上下文集合重叠的边缘节点，所以要有K>r1，这些节点承接了两部分，我们使用另一个GNN学习这些节点的表示，然后再平均起来作为节点v的上下文信息，如上图所示。这样训练目标就是一个二分类问题，判断两个表示是否来自同一个中心节点。

![image-20200408141050716](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2020%5D%20Pre-training%20graph%20neural%20networks/image-20200408141050716.png?raw=true)

​	这样的思想与NLP预训练词向量中的Word2Vec非常相似。

​	第二种方法更加直接，即我们MASK掉图中的一些特征，然后用GNN学习其邻居的属性特征，再来预测这个节点的特征。同理还可以MASK掉边的属性

![image-20200408141606245](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2020%5D%20Pre-training%20graph%20neural%20networks/image-20200408141606245.png?raw=true)

### GRAPH-LEVEL PRE-TRAINING

​	在图级别的预训练中同样存在两个可行的任务，预测整个图的属性和预测整个图的结构。

​	预测图的属性是一个有监督的任务，需要一系列的图以及其标签，这在化学和药物领域通常比较容易获得，比如预测一个蛋白质图的功能。但是这样的有监督任务通常会降低下游任务的迁移效果，因为这样的标签可能与下游任务并不一致，但是我们发现首先经过上面的节点级别预训练之后，这样的负迁移会有一定的缓解。

​	第二个可行的任务是图结构的预测，判断两个图的结构相似性，但是其真实标签难以界定，所以我们目前没有采用这种方法。

## Experiments

### Datasets

​	两个化学领域的分子图数据集，两个生物领域的蛋白质图数据集

- ​	ZINC15 
-   ChEMBL
- PPI
- protein ego-network

### Downstream

​	分子化合物预测：MoleculeNet

​	蛋白质性质预测：PPI网络中，四十个下游二分类任务。

### Result

​	![image-20200408190157313](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2020%5D%20Pre-training%20graph%20neural%20networks/image-20200408190157313.png?raw=true)

## 个人见解

​	本文是ICLR上面少有的没有出现公式的文章（正文），正是如此，其并没有提出新的模型，而提出的是一种策略，即给出了数据集以及定义了任务，并说明了使用GNN模型的方式，其思想看起来非常高大，要做GNN上的预训练模型，也确实是参考了NLP上的一些方法，但是其致命点在于没有完全拜托有监督训练，我们知道出色的预训练模型不应当依赖于任何的下游任务，但是这里在整图属性预测上用到了图的功能标签，而测试的下游任务又恰恰是对整体标签的预测，给人一种投机取巧的感觉，虽然确实提出了一种预训练的方法，但是我觉得并不能成为图领域的BERT那样的地位。