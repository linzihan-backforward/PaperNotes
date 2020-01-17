# Self-Attention Graph Pooling

------

## Motivation

​	类比CNN在结构化的数据上的出色表现，将卷积的思想用于非结构化的图数据中便得到了GCN模型，虽然这种迁移展现出了非常好的效果，但是在图数据上如何进行Pooling一直没有很好的解决，为了与图上的卷积相配合达到好的特征提起的效果，同时降低模型参数提高重用性，本文新提出一种使用Attention的Pooling方法能够在合理的时空复杂度下得到一个特征和结果的层次化表示。

## Model

​	在介绍模型之前，我们先来说一下图上的Pooling究竟是要做什么，其实英语中的Pooling的意思更容易理解：down sampling ，下采样，就是从很大量的集合中采样出其中有代表性的，而丢掉其余冗余的，在CNN中通过Pooling来提取出一小块区域中的代表性特征便是其能够降低模型参数的关键，那么在图数据中，这种下采样就迁移到了节点的头上，通过选择出某些节点来达到减小节点和边的数量的目的，同时最大可能的保留图的结构和节点的属性，这就是我们的目标。

​	为了实现这个目的，我们希望为每一个节点学习到一个权重，然后根据此权重来进行Pooling，这就与Attention的思想相契合了，具体地，根据一个图的结构和特征自己学习出权重，需要选择一个self-attention的函数，因为GNN是一个很好的图上的方法，所以这里同样使用一个GNN来作为此权重函数。

![image-20200116185242238](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200116185242238.png?raw=true)

​	这个公式与传统的GCN非常类似，区别在于输出Z是一个N×1的向量，也即我们需要的节点对应的权重。得到权重后，我们就可以根据Zi的大小和采样率k来选择其中的topK个。

![image-20200116185529871](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200116185529871.png?raw=true)

​	根据这个mask从原图中选择出采样得到的节点，将其余节点和连接全部删去，得到Xout和Aout分别对应输出图的特征矩阵和邻接矩阵，这样就完成了一个图上的Pooling操作。

​	为了更具一般化，我们可以将最原始的GCN的操作统一为各种的GNN运算，甚至多层GNN或者多个GNN的平均值，如下面公式中的几种计算方法在实验中均用到。

![image-20200117102533982](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117102533982.png?raw=true)![image-20200117102548308](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117102548308.png?raw=true)![image-20200117102557905](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117102557905.png?raw=true)

​	有了这种图上的池化操作之后我们能够来干什么呢？因为其作用是保留图中的有代表性的节点，所以一种使用池化的模型便是将图经过池化操作后输出的节点特征进行聚合得到图的特征，然后再用此特征对图进行分类，为了与之前的一些方法中的模型结构相一致，我们只使用最基础的GCN方法来实现卷积求权重的作用，在图特征聚合操作中我们将图中所有节点的特征求平均再与最大特征进行连接，如下面公式

![image-20200117103938216](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117103938216.png?raw=true)

​	Pooling与GCN相结合可以形成深层次的网络结构，与之前的工作相一致，我们仅使用了其中的两种来验证我们提出的Pooling方法的有效性，这两种结构分别见下图。

![image-20200117104123092](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117104123092.png?raw=true)

## Experiments

### Datasets

- **D&D**  一个蛋白质网络数据集，节点为氨基酸，预测此蛋白质是否为有用的酶
- **PROTEINS**  同样为一个蛋白质网络数据集
- **NCI**  化学分子数据集，节点为原子，边为化学键，预测此分子是否有抗癌活性
- **FRANKENSTEIN**  化学分子数据集，预测此分子是否为诱变剂

### Baselines

- **Set2Set**  一个使用LSTM来做集合简化的方法，不仅适用于图
- **SortPool** 使用排序来进行Pooling的方法
- **DiffPool** 第一个端到端可训练的图上Pooling的方法
- **gPool** 与我们方法非常相似的一种使用排名进行Pooling的方法，但没有考虑拓扑信息

### Results

![image-20200117142236300](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Self-Attention%20Graph%20Pooling/image-20200117142236300.png?raw=true)

## 个人见解

​	这个模型的目标是进行图的分类，但是与其他直接使用邻接矩阵的方法不同的是其借鉴了CV中对图片的Pooling处理的思想，这个想法可能不是本文首创的，但是这是一个很新颖很有意思的想法，在实现Pooling过程中采用的是GCN的方法，所以严格来说这应该算是GCN的一个应用，模型上并没有太大的创新。这种Pooling的方法可以跟GNN结合起来从而让其真正像图上的CNN一样的结构，这方面的工作还比较少，在节点的embedding已经被玩烂了之后，这种图上的Pooling方法我认为可能会是接下来一个GNN研究的点。