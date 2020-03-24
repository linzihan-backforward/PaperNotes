# Community-Based Question Answering via Heterogeneous Social Network Learning

------

## Motivation

​	在CQA的数据中，稀疏性是一个影响很大的因素，很多问题和用户都仅仅出现一两次，导致矩阵分解类方法在使用上的困难。除此之外，很多基于内容的方法使用启发式的文本特征来匹配问题和回答，其没有考虑语序也就无法挖掘出文本深层次的语义信息。为了解决CQA中的问题路由，本文考虑融入用户之间的社交信息来缓解稀疏性，同时提出一种图上游走的方法来同时建模文本语义、用户关系和回答的相对质量。

## Model

![image-20200323175530320](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BAAAI2016%5D%20Community-Based%20Question%20Answering%20via%20Heterogeneous%20Social%20Network%20Learning/image-20200323175530320.png?raw=true)

​	在CQA中，一个关键的部件便是文本编码器，在本文中，使用LSTM作为Encoder来处理所有的问题和回答的文本，对于文本中的多个句子，分别使用LSTM得到其最后一个特征向量之后再进行MaxPooling。

​	在上图的异质图中，包含两个子图，一个是问题、回答、回答者形成的CQA网络，另外一个是用户和用户之前形成的社交网络，为了将这两部分网络一起建模，本文中使用DeepWalk的方法，但是基础的DeepWalk无法刻画文本信息，所以我们修改其结构和损失来让问题和回答的节点向量分别由其文本经过LSTM得到，同时对于在同一个上下文内的质量不同的回答，让损失考虑其相对得分。

![image-20200323184338619](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BAAAI2016%5D%20Community-Based%20Question%20Answering%20via%20Heterogeneous%20Social%20Network%20Learning/image-20200323184338619.png?raw=true)

![image-20200323184349809](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BAAAI2016%5D%20Community-Based%20Question%20Answering%20via%20Heterogeneous%20Social%20Network%20Learning/image-20200323184349809.png?raw=true)

## Experiments

### Datasets

​	来自于Quora和Twitter爬取的数据，由Twitter ID作为连接，包括252k问题、381k回答、67k用户

### Baselines

- **BOW** 将问题和回答的文本平均得到向量，计算相似度来排序
- **LDA** 将文本映射到定义好的隐藏语义空间，再计算匹配分数进行推荐
- **Doc2Vec**  分布式的BOW模型
- **DeepWalk** 仅仅使用图的结构信息而不使用文本
- **CNTN** 使用卷积张量网络和动态池化层来建模问题和回答
- **S-matrix** 使用相似度矩阵来建模问题和回答的交互

### Results

![image-20200324091437383](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BAAAI2016%5D%20Community-Based%20Question%20Answering%20via%20Heterogeneous%20Social%20Network%20Learning/image-20200324091405133.png?raw=true)

## 个人见解

​	本文的思想是将社交网络的连接信息与文本特征的学习放到了一起，在好几篇方法和内容基本相同的文章中，这一篇应该算是时间最早的，所以就将其当作开端来写一下。首先其将社交网络中用户之间的关注信拿过来加强用户之间的连接，这样的思想是没有问题的，但是有关注关系的用户就可能回答相似的问题吗，或者说我关注的人回答了一个问题，就说明我回答的可能性比别人大吗，还是缺少一些直观概念上的支撑。另外，本文将Skip-Gram向异质路径拓展，对于不同类型的节点使用不同的损失函数，这一点应该是独创的。