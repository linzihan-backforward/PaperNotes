# Graph Neural Networks for Social Recommendation

------

## Motivation

​	在社交推荐系统中，我们可以拥有用户的社交网络信息（用户之间的家人、朋友关系）和用户的历史打分信息（对某一商品给出评分），这两类信息均可以天然的表示为图的关系，如何将图结构信息和用户、商品的语义信息统一到一个表示中便是本文的目标，图神经网络（GNN）天然具有聚合图中临近节点信息的功能，所以考虑使用GNN作为技术的基础，但是传统的GNN在这个问题上又存在局限：用户关系网络中包含不同的关系（不仅仅关注、被关注），用户商品交互网络中打分信息非常重要、用户同时在两个网络中出现。本文便是对GNN进行了拓展使得能够解决上述局限。

## Graph that users involved

​	上面提到，在这种推荐系统中，不仅有用户历史上对商品的打分，还有用户之前关系，利用这两类信息，可以分别构建两个图，即用户-商品交互图（下文中以U-I代指）和用户-用户关系图（下文中以U-U代指）。在U-I图中用户和之前评价过的商品连一条边，边权为用户的打分值。在U-U图中，存在关系的用户之前连接边，这里为无权图，不同的关系放在模型中学习。

## Model

​	在推荐系统中，核心问题是得到用户表示（user embedding）和商品表示（item embedding）本模型也是基于这两个部分将整个模型划分为两个部分，分别得到这两个表示，然后通过打分模块得到最后的交互分支，下面分别简单介绍这几个模块。

![image-20191224102905487](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Graph%20Neural%20Networks%20for%20Social%20Recommendation/image-20191224102905487.png?raw=true)

### User Modeling

​	用户这个实体是包含在两个图中的，而想要得到包含两种关系的用户表示必然要同时在两个图上学习，在这里采用的实际上还是简单的Spilt and Concatenation 的思想，即分别在独立的两个图上进行，然后把结果连接得到最终的用户表示，这两个图上的模型基本上是一致的，这里以U-I图为代表讲述，U-U图大部省略。

#### Item Aggregation

​	首先使用常见的低维稠密表示的思想为每一个实体随机初始化一个embedding：用户i的embedding **pi**，商品i的embedding **qi**，除此之外为了使用打分信息，将分值也进行embedding ，对于分值为r的边，得到 **er**（五星评分则分别得到5个vector）。

​	有了这些之后要得到的便是高层的用户表示 **hIi**，采用的方法是聚合图上用户节点i周围的商品节点的信息。对于一条用户i连出的边，将 **q** 和 **e** 通过一个MLP之后得到边的表示 **x** 即：![image-20191224104216040](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Graph%20Neural%20Networks%20for%20Social%20Recommendation/image-20191224104216040.png?raw=true) 得到之后一个简单的方法便是把用户i的每一条边的 **x**直接平均实现聚合，但这丢失了不同商品之前的重要性差异，所以使用Attention的办法来为每条边学习其权重，同样使用一个MLP，输入为用户 **pi** 和 **xia** 输出为此边的未归一化权重，再进行softmax得到权重，最后加权平均每一条边再经过MLP以得到高层表示 **hli**![image-20191224151213548](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Graph%20Neural%20Networks%20for%20Social%20Recommendation/image-20191224104737423.png?raw=true)

#### Social Aggregation

​	采用同样的思路用在U-U图上，聚合节点邻居信息，对于所有的邻居用户节点，加权平均其高层表示 **hlo**，得到社交层面的用户高层表示 **hsi**，使用的是同样的Attention的方法，不再赘述。![image-20191224110413518](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Graph%20Neural%20Networks%20for%20Social%20Recommendation/image-20191224110413518.png?raw=true)

#### User Latent Factor

​	使用上两个模块得到的结果连接后经过MLP，得到最终的用户表示 **hi**

### Item Model

​	商品只出现在U-I图中，所以也就只用得到一个商品表示 **zj** 方法依然是在图中聚合周围的用户节点，与Item Aggregation 部分完全一致，只是用户节点和商品节点的地位互换。

### Prediction

​	有了用户表示 **hi** 商品表示 **zj**  后将两个vector 连接送入MLP，最后一层为一个实数输出值作为评分预测，这一步与其他工作一样，没有什么出彩之处。

### Optimization

​	整个网络为端到端的设计，输出为一个 u，i 对和其分别在两个网络的邻居，输出为预测评分，使用L2距离作为Loss，RMSprop作为优化器进行训练。



## Experiments

本文的代码开源： https://github.com/wenqifan03/GraphRec-WWW19

### baselines

- PMF ：单纯使用评分矩阵进行矩阵分解的传统推荐算法

- SoRec：使用两个矩阵进行共同分解

- SoReg：使用社交关系信息作为正则项对评分矩阵进行分解

- SocialMF：考虑了信任在社交网络中的传播关系的矩阵分解

- TrustMF：将用户表示分为两个空间，分别进行矩阵分解

- NeuMF：使用神经网络架构对评分矩阵进行矩阵分解

- DeepSoR：使用神经网络学习社交关系信息并用在矩阵分解中

- GCMC+SN：使用node2vec对社交图做嵌入得到用户的额外信息

### result

  ![image-20191224145244317](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Graph%20Neural%20Networks%20for%20Social%20Recommendation/image-20191224145244317.png?raw=true)