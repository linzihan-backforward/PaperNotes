# IntentGC: a Scalable Graph Convolution Framework Fusing Heterogeneous Information for Recommendation

------

## Motivation

​	使用图结构数据来构建推荐模型近几年来发展非常迅速。有很多的方法直接使用显示的交互信息来建图，但是实际中这种直接的信息（如用户对商品的点击）是非常稀疏的，所以有很多的研究考虑将额外的辅助信息纳入模型。如社交网络关系、商品特征信息等，但是已有的这些工作都仅仅使用了一类的额外辅助信息而忽略了多种不同类型的额外信息之间的共同作用。

![image-20200109085738112](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109085738112.png?raw=true)

​	如上图所示，左边为比较单一的交互图，在用户端仅使用了社交关系，商品端仅使用使用共现关系，而我们提出的方法能够处理右边这种包含各类复杂关系的交互图，通过合并考虑各类的辅助信息，能够进一步提升图推荐方法的效果。进一步使用基于GCN的改进模型，在能够刻画多种关系的基础上进一步简化模型的计算量，提升模型速度。

## Model

​	在线购物场景中的推荐任务的主要目标是预测用户对商品的选择，即根据用户历史的交互记录来预测其下一个可能的交互商品，则在得到如上图中右侧的图之后，任务可以抽象为对（U，I）之间进行连接预测的问题，这样我们其实可以将这个复杂的异质图进行分类，节点（U，I）为主要节点，其他节点（如：搜索词、店铺、品牌）为辅助节点，（U，I）之间的边为关键的有标签的边，定义为Elabel，其余为辅助边。有了这样的定义之后，便可以分步来介绍处理这种复杂异质图的推荐模型了，模型分为三个步骤：对原始图进行转换，使用轻量级卷积网络学习，双图卷积分别建模用户和商品。下面依次进行介绍。

### Network Translation

​	直接处理原始的这种复杂的异质图是非常困难的，其各种不同的连接关系不仅带来了更多的信息，也带来了更多的计算量，为了能够尽可能地使得模型简单化来适应更大规模的数据，我们首先将原始的异质图进行转化，采用与之前工作中相似的思路，因为我们关注的重点为（U，I），所以我们分别转化为（U，U）图和（I，I）图，转化的方式为使用它们的二阶关系，即对于每一对（ui，uj）如果其有公共的其它类型的节点邻居，则我们将（ui，uj之间进行连边），边权为此类型的公共节点的数量。注意，同一对（ui，uj）之间可能存在多种边，即多种的辅助节点可能均会导致其相连关系，所以我们假设图中共有R类的辅助节点，那么每一个u应当有R种连接关系，R种连接参数，R种邻居。使用同样的方法对U和I进行建模，得到的两个图只包含U或者I但包含各种不同的连接关系。

### Faster Convolutional Network：IntentNet

​	得到了转化之后的图之后，我们使用GCN的思路，在图上使用一种简化的卷积操作来刻画多种的连接关系，同时降低计算量。

​	传统的GCN中，每一层卷积包含两个部分：邻居信息的聚合和特征的非线性映射，其中在第二部中存在需要学习的模型参数，这一步的映射是通过一个全连接层来实现的，即如果输入的向量维度为d，则本操作所需要的计算量为d×d，我们受到CV中对于卷积核拆分的启发，将这个映射操作中原本一个d×d的映射矩阵替换为两个小矩阵（d×L+L×d），这样因为在实现时L<<d，所以这一步的计算量就变为了d×L，可以说是下降了一个量级。那么这样操作有什么理论含义嘛。

​	原本的映射操作中，输出的每一个维度都是由输入的所有维度计算而来，我们把每一个维度认为是一个特征的话，相当于所有的特征之间都进行了交互，而我们认为有一些特征之间的交互是没有意义的，如节点的年龄与周围邻居的评分特征之间的交互不会产生有用的信息，所以我们限定同样的维度之间进行交互，这里的L相当于L个过滤器，每个过滤器分别学习一种交互方式，然后再聚合，形式化的对比如下：

![image-20200109112509241](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109112509241.png?raw=true)

​	进行了这样的改造之后，我们的卷积层运算的更快了，但是现在我们的图中节点之间存在各种不同类型的连接，每种连接作用可能不同，所以我们将每一种连接形成的邻居分开进行聚合，然后再作为不同的邻居向量来进行映射，公式如下：

​	![image-20200109113317237](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109113317237.png?raw=true)

​	将这种简化的卷积层进行堆叠得到最后的模型，如下图所示：

![image-20200109113449062](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109113449062.png?raw=true)

### Dual Graph Convolution

​	在（U，I）所形成的图上分别进行GCN之后，可以得到用户和商品所对应的表示，接下来的方法就跟大多数的方法一样了，使用负采样同时优化Hinge Loss即可，这里的双GCN除了所处理的图不同之外其余的所有操作都是相同的。

![image-20200109120521029](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109120521029.png?raw=true)

## Experiments

### Datasets

- 私有的Taobao数据集，来自其9天的数据
- 公开的Amazon数据集

### Baselines

- **DeepWalk**：直接在I-I同构图上使用DeepWalk
- **GraphSage** ：GCN中用来做推荐的模型，在I-I图上做相似推荐
- **DSPR** ：基于DSSM的方法，已经被多个商品平台使用
- **Metapath2vec++** ：异质图嵌入的方法，将其用在转换之后的图上
- **BiNE** ：仅仅使用一种额外信息来帮助建模用户和商品的方法
- **IntentGC（Single）** ：使用本文提出的模型，但是仅加入一种的额外辅助信息。

### Results

![image-20200109142300527](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20IntentGC_%20a%20Scalable%20Graph%20Convolution%20Framework%20Fusing%20Heterogeneous%20Information%20for%20Recommendation/image-20200109142300527.png?raw=true)

## 个人见解

​	随着GCN的各种变形都已经出的差不多，19年将GCN进行生产场景落地的方法也多了起来，这篇文章虽然也是在Taobao场景下的大数据集GCN推荐方法，但是和stanford那个PinSage还是感觉非常不一样的，哪个模型的思路是采用一些实现和部署上的trick来加快训练过程，而这个是通过简化模型结构来加快，可能更加有一点学术味。这篇文章思路上还是与内容推荐相一致的，将用户和商品来进行分别建模，两个平行的Model，在最后打分上进行交互，这种思路已经是比较常用了的，效果上也可以，结构上还非常简单，但是放到图上感觉就有点没有完全使用图的连接信息的概念，将整个图一分为二考虑，丢掉了最关键的边的信息。这个关于GCN计算上的改进确实减少了很多的计算量，但是感觉理论上比较单薄，相对应的特征进行交互也不是那么说得通，但是效果上确实说得过去，再加上这是专注于工业环境的模型，所以也不是很大的问题。