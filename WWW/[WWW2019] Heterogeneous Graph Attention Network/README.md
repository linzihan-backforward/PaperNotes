# Heterogeneous Graph Attention Network

------

## Motivation

​	GNN作为图数据处理的一大利器，在诞生之后便迅速成为了研究的热点，尤其最近的GAT将attention机制用于聚合函数上更是拓展了其应用的范围。但是在之前的GNN相关工作中都没有充分考虑异质图，其会包含各类不同类型的节点和连接。所以在本工作中我们将GAT拓展到了异质图中，通过两层的注意力机制，对节点的不同类型邻居分别处理，取得了非常好的效果。

## Model

​	整体的模型分为两个部分，第一部分使用meta-path找出节点周围的不同的邻居节点，第二部分将各个meta-path的信息进行聚合，两步均使用注意力机制，形成一个层次化的注意力模型，如下图。

![image-20200220114741015](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220114741015.png?raw=true)

### Node-level Attention

​	在异质图中，不同类型的节点具有的特征是不同的，不能够直接在一个特征空间进行计算，所以第一步应当是将所有的节点特征向量转换到同一空间中。

![image-20200220114953980](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220114953980.png?raw=true)

其中M是转换矩阵，共有|节点类型|个，hi为原始特征。转换之后我们需要定义节点周围的邻居，对于一个预先定义的meta-path ф，以当前节点为起点，严格按照ф进行遍历，得到的所有终止节点便组成了当前节点在ф上的邻居集合，这第一层的attention便是要找到这个集合中各个邻居的权重。

![image-20200220115530116](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220115530116.png?raw=true)

这里需要注意的是att（）函数应当是不对成的，因为两个节点类型不同，地位不等。如使用softmax方式实现如下。

![image-20200220115824571](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220115936372.png?raw=true)

​	在计算出当前meta-path ф对应的邻居集合的权重后，使用加权聚合得到节点i在meta-path ф上的特征表示。为了增强这一步的刻画能力，重复K次上面的步骤便得到K头的注意力，最终的表示为K头的输出相连接。

​	![image-20200220120019860](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220120019860.png?raw=true)

​	在完成了这一层的attention后，对于给定的所有P个meta-path，每一个节点都有P个对应的特征表示，下一层的attention便是将这P个聚合为一个作为节点的最终表示。

### Semantic-level Attention

​	同样的，我们需要使用attention机制来得到每一个meta-path所对应的权重

![image-20200220142622484](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220142622484.png?raw=true)

为了统一衡量每一个meta-path的作用，我们使用一个MLP+点乘相似度的方式得到每一个Z的得分。

![image-20200220143025356](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220143025356.png?raw=true)

softmax得到权重

![image-20200220143112931](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220143112931.png?raw=true)

加权求和得到最终结果

![image-20200220143132819](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220143132819.png?raw=true)

​	最终Z作为每一个节点的表示，根据不同的任务目标可以使用不同的Loss来进行反向传播。

## Experiments

### Datasets

- ​	**DBLP** 包含：paper(p)、author（A）conference（C）term（T）选择meta-path：APA、APCPA、APTPA
- **ACM** 包括：papers（p）、authors（A）subjects（s）选择meta-path：PAP、PSP
- **IMDB** 包括：movies（M）、actors（A）、directors（D）选择meta-path：MAM、MDM

### Baselines：

- **DeepWalk**
- **ESim**
- **metapath2vec**
- **HERec**
- **GCN**
- **GAT**

### Results

![image-20200220145029549](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Heterogeneous%20Graph%20Attention%20Network/image-20200220145029549.png?raw=true)

## 个人见解

​	本文是将GAT方法应用到异质图的一个拓展，通过将原本一层的attention聚合拆分为两层来分别对不同类型的邻居进行处理，这个问题本身没有让人兴奋的地方，但是在实现的思路上引入了meta-path这个中间媒介还是与基础的思路有了一点区别，不是把一阶邻居按照类别分类，而是直接聚合遥远的高阶邻居，这种方法与传统的GNN的思想相悖，但是又融合了早期NRLdeepwalk的思想，这种方法能够使得模型收敛也确实让人没想到，也给了一些老的网络表示方法与新的聚合方式结合的可能，同时这个异质图的工作能够作为多个方面应用的baseline