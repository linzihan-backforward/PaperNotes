# Heterogeneous Graph Neural Network

------

## Motivation

​	异质的网络相比同质的网络来说更加普遍，更加有用也同时更加难以处理，已有的工作中有很多在同质网络基础上向异质网络推广的工作，但这些工作仅仅采用了简单的处理方式如metapath2Vec，其仅仅考虑了节点之间的异质性，但是节点可能同时伴随有异质的属性和特征，将节点的特征纳入节点表示之中事GNN所擅长的工作，但是不论是直推式的GNN还是归纳式的GAT、GraphSAGE，其都是以同质网络为基础的，所以我们考虑结合GNN的属性处理优势与DeepWalk-based方法在异质图上的优势来同时处理具有节点间属性异质性的异质图，将不同类型的属性信息（数值、文本、图像）统一处理，并尝试聚合不同类型的邻居节点，主要的流程为：邻居节点采样、不同类型特征提取、邻居聚合。分别对应下图中三个部分。

![image-20200114094350453](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114094350453.png?raw=true)

## Model

​	对应上面的三个阶段，把模型也分成三个功能进行介绍。

### Sampling Heterogeneous Neighbors

​	在GNN方法中，对节点邻居的采样往往直接选择与其直接相连的邻居，但是这样的操作在异构图中会产生一些问题：不同类型的节点之间可能不会直接相连，一些密集连接的中心节点可能会被少数的噪声节点干扰，连接很少的冷启动节点不能很好的找到直接相连的邻居，具有不同类型特征的邻居不能直接聚合。

​	为了解决在异质图上的上面的局限，我们在采样阶段使用DeepWalk的思想，从节点u开始进行概率返回的随机游走，每一步沿着图中的边选择下一个连接节点或者以概率p返回起点，将得到的整个随机游走的序列作为节点u的邻居，这样理论中u的邻居就包含了所有类型的节点，且集合的大小也能够固定，接下来为了方便下一步对不同类型的节点进行处理，我们还需要将随机游走集合中的节点按照节点类型进行归类，每一类的节点我们按照其在此集合中的出现频率取前topk个，这样对于一个包含T个不同类型节点的图，每一个图中的节点u就具有 Topkt×T个邻居节点，且按照类型进行了归类。

### Encoding Heterogeneous Contents

​	在采样得到邻居节点之后，第一步是将邻居节点中不同类型的内容信息（属性）统一建模为一个固定长度的向量。

​	对于文本、图片这种无法直接处理的属性类型，采用预训练的方式将其处理为一个数值向量xi，如使用Par2Vec来预训练文本，CNN来预训练图片，在得到其可计算的表示xi之后，与之前模型中直接连接作为本节点的属性向量不同，我们使用双向的LSTM（Bi-LSTM）来刻画属性之间的深层的的特征交互，即我们将一个节点的所有属性向量通过LSTM，将LSTM的输出特征作为所有属性的高层表示，也即节点的特征向量。

![image-20200114105047074](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114105047074.png?raw=true)

​	式子中每一个特征通过一个不同的FC来映射到一个空间中，然后再依次通过LSTM，将双向的表示连接得到结果，每一个特征对应的LSTM输出平均之后得到节点的表示。形象化的表示如下图。

![image-20200114105531290](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114105531290.png?raw=true)

### Aggregating Heterogeneous Neighbors

​	在上面两步得到了邻居集合以及节点的表示之后下面就要进行邻居节点的聚合，因为涉及到不同类型的节点，所以聚合的过程分为两步。

​	首先是同类型节点的聚合，对于同属于一类的节点，其包含的特征是一样的、表达的含义是相似的，所以使用一个无序的聚合函数来处理即可，借鉴于之前的工作，与特征聚合相同，我们依旧使用Bi-LSTM来近似聚合操作。

![image-20200114110506918](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114110506918.png?raw=true)

​	这样此结构与上面完全相同，除了输入输出不同，所以实现时结构上可以最大化的重用。

​	然后是不同类型节点之间的聚合，考虑到不同类型所处的角色不同，贡献不同，所以使用Attention机制来作为聚合函数，即利用本节点的表示作为查询，所有T个类型的表示作为Key来计算权重再进行加权和。

![image-20200114111153834](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114111153834.png?raw=true)

​	三个模块介绍完之后整个模型的流程如下图。

![image-20200114111324443](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114111324443.png?raw=true)

​	其中NN-1、2、3分别对应特征聚合、同类节点聚合、类间邻居节点聚合的神经网络。

### Objective and Model Training

​	前向传播的到节点的表示之后，需要定义目标Loss来进行后向传播，为了模型同时适用于多种下游任务，这里采用无监督的Loss，即与DeelWalk方法中使用负采样的目标函数保持一致。对于每一个节点，最大化其与邻居节点的点积同时最小化其与负样本的点积。

![image-20200114142030052](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114142030052.png?raw=true)

## Experiments

​	作者开源了Pytorch实现的实验代码：https://github.com/chuxuzhang/KDD2019_HetGNN

### Datasets

​	为了测试不同的下游任务，如连接预测、节点分类等。使用了两类异质图数据集：学术网络和评价网络

- 学术网络：AMiner 中根据时间切分出两个不同的数据集
- 评价网络：Amazon 数据集中的Movies和CDs分类

### Baselines

- ​	无监督无附加特征的**metapath2vec**
- ​    同质属性图：**ASNE**  、**SHNE** 
- ​    GNN方法：**GraphSAGE**、**GAT**

### Results

![image-20200114143551377](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Heterogeneous%20Graph%20Neural%20Network/image-20200114143551377.png?raw=true)

## 个人见解

​	这篇文章可以算是一个标准的GNN的变形，其最主要的创新点在于将GCN中对邻居节点的聚合操作进一步泛化，将原本半固定的聚合函数AGG（）彻底使用神经网络（LSTM）这个非固定的函数进行替代，以此实现了多类型特征聚合的功能，同时将DeepWalk这种无监督的方法和GCN这种有监督的方法进行了一个结合，使得这种变形模型能够比原模型实用性更广，但是这种泛化模型的方法还是将不同类型的节点特征进行了预训练而并没有100%的做到端到端的归纳式模型，另外其复杂结构在同质图上是否同样有效没有被说明，一种比LSTM更好的方式显然是天生顺序不敏感切并行的Attention。