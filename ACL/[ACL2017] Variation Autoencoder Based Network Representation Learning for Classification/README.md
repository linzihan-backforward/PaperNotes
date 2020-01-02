# Variation Autoencoder Based Network Representation Learning for Classification

------

## Motivation

​	在网络表示学习的任务中，我们要将网络中的节点映射到连续的实数空间，已有的方法没有使用节点所包括的丰富的额外特征，且仅仅刻画了一个小范围内的子图的网络结构，这些方法本质上都是使用的基于邻域假设的skip-gram模型，所以无法对高阶的关系和全局的网络结构进行刻画，本工作针对这个目标，提出了一种完全不同的节点嵌入的模型架构，使用变分自编码器（VAE）来同时建模节点所包含的文本信息和全局网络结构信息。

## Model

​	我们定义此模型面向的图是具有额外的文本信息的图，即对于每一个节点vi，由一段额外的文本信息ci，对于最终的节点表示，不仅需要将网络结构的信息嵌入还需要将文本ci嵌入。

​	因为模型中用到了变分自编码器（VAE），在这里简单介绍一下，VAE是传统的自编码器（AE）的一种变形，在AE中Encoder的作用是将输入编码到一个维度更小的空间中，而解码器负责将其还原到原始输入，这样通过让解码结果与输入想接近，我们就能近似得到输入的低维度表示，即嵌入。但是传统的AE存在问题，即我们只能对已有的数据进行嵌入而无法生产新的数据，即有用的只是编码器模块，如果我们想用解码器模块怎么办呢？这就需要VAE了，VAE在AE的基础上加入了一个先验，即编码之后的特征应当是在高斯分布空间的，编码器将所有的输入编码到一个高斯分布，那么我们只要在这个高斯分布中进行采样应当都能够使用解码器得到新的数据，即实现新数据的生产。基于这样的先验，我们的编码模块只需要输出一个均值和方差就能够代表输入的高斯分布，则编码器部分的Loss可以定义为输出高斯分布与标准高斯分布之间的KL散度。对于解码器，我们需要为方差添加上一个随机的噪声，然后输入进解码器，使得输出与输入相接近。

​	了解了VAE之后，我们的模型其实就非常简单了，首先对于每个节点的文本信息部分，使用现成的doc2vec来将其嵌入到一个实数向量中方便之后处理。

​	对于每一个节点我们使用其邻接向量来代表其图中的连接关系，即对于每一个节点i ，ai为一个1×n的向量，其中每一个值为1代表节点i与其有边相连，0代表没有。对于每一个节点将ai与doc2vec之后得到的向量ui进行拼接形成xi，xi作为VAE的输出，编码器的输出我们修改为4个输出头，分别对应ui和ai的均值和方差，解码器与传统的VAE一样。那么我们节点的表示向量怎么得到呢？很简单，使用一个线性函数分别计算两对均值方差，然后连接即可，整个运算过程如下图所示。

![image-20200102100509855](https://github.com/linzihan-backforward/PaperNotes/blob/master/ACL/%5BACL2017%5D%20Variation%20Autoencoder%20Based%20Network%20Representation%20Learning%20for%20Classification/image-20200102100509855.png?raw=true)

​	这里的Loss与传统的VAE的Loss一样，KL散度和解码器的还原误差。

![image-20200102100643295](https://github.com/linzihan-backforward/PaperNotes/blob/master/ACL/%5BACL2017%5D%20Variation%20Autoencoder%20Based%20Network%20Representation%20Learning%20for%20Classification/image-20200102100643295.png?raw=true)

## Experiments

### datasets

​	使用了两个数据集，均为论文的引用网络形成的数据集：（1）CiteSeerM10 http://citeseerx.ist.psu.edu/

​                                                                                              （2）DBLP   http://arnetminer.org/citation

### baselines

- **One-Hot**  直接使用节点的邻接向量送入分类器。

- **DeepWalk**  无监督的网络嵌入方法，只使用网络结构信息。

- **Node2Vec** 同样无监督的方法，可以看作DeepWalk的升级版

- **Doc2Vec** 仅仅使用节点的文本信息，将文本向量作为节点嵌入。

- **DW+D2V** 将DeepWalk结果和Doc2Vec的结构相连接。

- **TADW**  使用矩阵分解的方法将文本信息融入了DeepWalk方法中。

- **TriDNR**  有监督的方法，同时使用了节点的标签、网络结构、文本信息。

### Results

![image-20200102102454534](https://github.com/linzihan-backforward/PaperNotes/blob/master/ACL/%5BACL2017%5D%20Variation%20Autoencoder%20Based%20Network%20Representation%20Learning%20for%20Classification/image-20200102102454534.png?raw=true)

## 个人见解

​	这篇文章的模型确实是其Motivation中提到的那样与之前的基于在图上随机游走的方法不一样，也确实是对整个图来建模而不是对一个区域子图。但是除了确实满足这几点之外，真的找不到其他的有创新或者给人启发的地方，使用的基本上是原始的VAE模型，对高阶关系的刻画是将稀疏的one-hot矩阵作为输入，将文本信息合并的方法也是使用现成的doc2vec之后来连接，如果是这样的话跟使用两个独立的VAE来编码两个输入有什么区别呢，反正学到的是两个独立的高斯分布，这种使用raw-feature+连接的方法是否能够有如此好的效果，是否好训练我感觉还有一定的疑问。这个模型的思想我觉得跟之后的一些使用meta-path路径矩阵的思想是一脉相承的：一个顶点和其他顶点之间的连接关系就包含了从低阶到高阶的所有信息，使用好这种连接关系的feature就可以直接刻画网络结构，而不用再使用随机游走来间接的得到结构信息。总体上质量不高，但是是一种思路上的尝试。