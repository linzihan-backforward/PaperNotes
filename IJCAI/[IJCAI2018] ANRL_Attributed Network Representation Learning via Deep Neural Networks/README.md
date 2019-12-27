# ANRL_ Attributed Network Representation Learning via Deep Neural Networks

------

## Motivation

​	在网络表示学习或者叫做图嵌入任务中，我们的目标是把一个图中的节点建模成一个低维的vector以此便可以进行下游的节点分类、连接预测等任务。现有的很多方法使用图结构来建模，即邻域假设：在图中相近位置的节点其表示应当相近。但是实际生产中图中往往还有其他很多的属性特征，比如社交网络中把人作为图中的节点，则性别、年龄、活跃程度都是可以作为节点附加属性的，而这个数据往往是非常重要的，如何同时将节点的属性和节点的连接关系共同编码到一个向量空间中，获得更好的节点表示便是本模型的目标问题。

## Model

### 	Notations and Formulation

​	对于一个图G=（V，E，X）V代表图中节点，E代表节点直接边的集合，X为所有的属性矩阵，其中xi为节点i对应的属性向量。

​	模型的目标是实现一个映射函数f，对于G中的每一个节点i∈V f(vi) --> yi，yi即为每个节点的低维表示。

### 	Neighbor Enhancement Autoencoder

​	为了编码节点的属性，自编码器模型进入了我们的视野，采用encoder将输入xi编码成任意维度的表示，再使用decoder将表示还原为原输入属性xi，这是一个最传统的低维建模方法，迁移到图上数据来，对其进行改造，将还原的目标从自身变为周边邻居的属性合，这样这个编码器便同时将周边节点的属性和连接关系考虑在内，但是该怎么定义周边邻居的属性合呢？文章给出了两种可行的方法：

1. 加权和，以边权作为权重将直接相连节点的属性向量x求和

2. 维度中位数，把权重向量的各个维度分开，每个维度的值是周围邻居节点相应维度的值的加权中位数，权值依然为边权

这样做的好处是什么呢：限制位置上相近的节点得到相似的编码结果，因为位置上相近则邻居集合也是相近的，那么重建的目标便是相近的，所以理论上编码结果也应当相近。这一模块使用二阶距离作为目标函数![image-20191227105746680](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227105746680.png?raw=true)

### Attribute-aware Skip-gram Model

​	有了上面的编码器就足够了吗，显然不是，再上面的模型中，我们只考虑了节点直接相连的的邻居节点，即一阶距离，而图中有相当多的信息包含在二阶甚至更高阶当中，所以还需要再加一个刻画高阶关系的模块，这时常用的Skip-gram又有了用武之地，怎么将属性信息添加到无监督的Skip-gram中呢，上面的编码器结构便可以在此重用，使用编码得到的低维表示来预测context 节点，这样充分的把编码部分送到一阶和高阶的模块中，采用一样的随机游走和负采样技术，不同的是原方法中的center vector变成了编码器输出encoder（xi），目标函数依旧是每一个window的log概率之和![image-20191227110008249](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227110008249.png?raw=true)

### joint Optimization

​	有了上面两个方法分别建模一阶和高阶关系后，便可以总结整个模型，包括一个编码器模块和两个解码重建模块，如下图。

![image-20191227110332576](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227110332576.png?raw=true)

​	整个模型的输入为当前节点的属性 xi 周围邻居的属性 xj和上下文节点，是一个端到端的优化模型，所以可以直接将两个模块的目标函数加和再加上参数的正则项得到整个模型的Loss Function。

![image-20191227110612586](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227110612586.png?raw=true)

​	当模型收敛之后，将编码器输出encoder(xi)作为模型结果yi输出。

## Experiments

### Datasets

​	这里作者使用了6个共三类数据集，还是相当丰富的，包括两个开源的网络数据集：Facebook社交网络数据集和Citeseer and Pubmed论文引用网络数据集，以及一个私有数据集：阿里巴巴用户行为网络数据集

### Baselines

​	同样作者将其余的baseline们分为了三类

- 只使用属性的方法。直接将节点的属性向量作为节点表示，或者自编码器降维后作为节点表示。

- 只使用网络结构的方法。丢弃节点的属性向量，使用Deep Walk、LINE等方法得到节点的表示。

- 同时考虑属性和结构的方法。这类方法是主要的对比目标。包含：AANE、SNE、SEANO等方法。

### Results

​	对于使用的6个数据集，三个用于连接预测任务，三个用于节点分类任务

![image-20191227135547939](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227135547939.png?raw=true)

![image-20191227135618152](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2018%5D%20ANRL_Attributed%20Network%20Representation%20Learning%20via%20Deep%20Neural%20Networks/image-20191227135618152.png?raw=true)

## 	个人见解

​		这篇文章的主要创新点在于编码、解码结构的设计，同时巧妙地重用结构进行多个任务的学习，不同的模块各司其值而又相互配合，巧妙之余又暗藏道理，实现了将任务目标拆解又组合的过程，非常有启发意义。实验的数据和对比也很充实，说服力很强。一个我觉得将来可以改进的地方我认为在邻居节点的特征聚合上，是不是可以有一种可学习的权重方式，引入类attention结构，这样聚合过程可以有一定的解释性。