# KGAT_ Knowledge Graph Attention Network for Recommendation

------

## Motivation

​	使用辅助信息而不仅仅是用户和商品的直接交互来建模属性和偏好已经是一个常用的加强推荐的策略，已有的方法中有很多方法将辅助信息编码之后使用相互独立的向量对来构建模型（FM类方法）但是用户之间和商品之间并非完全独立地，使用知识图谱中的额外链接关系可以将不同的商品连接到相同的属性上，这样打破独立的交互对，在图上建模推荐问题，这样就能够得到更好的商品建模效果，从而得到更优异的推荐结果。同时，相对于其他的使用知识图谱来进行推荐的方法，我们能够考虑高阶的关系而不仅仅是商品与其属性的一阶关系。

## Model

​	我们的模型是建立在图之上的方法，所以在介绍方法之前需要明确这个图（Collaborative Knowledge Graph）的定义，CKG，名字里面有KG那么肯定是在KG的基础上的，一个基础的KG是由三元组关系组成的（h，r，t）其中h或者t是代表的实体，我们能够将这些实体与商品对应起来，这样就可以把用户和商品之间的购买关系转变为（u，interact，e）其中u代表一个用户节点，interact代表一种新的关系，额代表与商品相对应的实体，这样就可以将知识图谱的有向图与用户商品之间的二部图放到一起形成我们定义的CKG。

​	在这个CKG中包含高阶的连接关系，如：用户-商品-实体-商品-用户这类，之前的方法并没有直接处理这种长距离的高阶关系，CF类的方法仅仅考虑了用户-商品-用户-商品这样的四步的关系，FM类的方法的核心思想是推荐具有相似属性的商品，所以仅仅考虑了用户-商品-属性-商品这样的四步关系，而我们的方法是这些方法的拓展，能够之间处理更高阶的连接关系。

![image-20200215090505407](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215090505407.png?raw=true)

​	我们在CKG上的模型可以分为三个部分，下面一一介绍。

​	在CKG上很大一部分是来自于KG，只是多了用户节点和交互关系，所以对CKG的嵌入其实与传统的KG的嵌入是基本一致的，模型的第一部分是embedding层，将整个CKG中的节点嵌入到向量空间中，使用的方法为TransR

![image-20200215091234604](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215091234604.png?raw=true)

所以在这一层中训练这个embedding会产生一个pair-wise的loss

![image-20200215091416959](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215091416959.png?raw=true)

在得到节点和关系的嵌入后，下面一层是embedding传播层，其作用便是传播长距离的关系，也是这个模型的主体。在CKG中，一个实体可能通过多种连接同多个其他类型的节点相连接，这可以理解为实体周围的一阶邻居会丰富实体的属性信息，或者说邻居的信息可以通过连接关系传递给当前节点，这其实就是GAT的思想，一层的GAT聚合一阶信息，多层则聚合多阶信息，类似GAT，定义对邻居集合的聚合方式为加权求和。

![image-20200215092526710](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215092526710.png?raw=true)

​	其中权重函数Π表示此邻居贡献信息的多少，其可以通过首尾节点的注意力打分求得。距离更近的节点权重应当更大

![image-20200215092652621](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215092652621.png?raw=true)

​	在得到邻居集合的传递信息后，需要对当前节点进行更新，在更新方式上其实有很多种的选择，相加之后经过一个FC

![image-20200215093330332](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215093330332.png?raw=true)

或者连接之后经过一个FC

![image-20200215093414701](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215093414701.png?raw=true)

或者同时考虑相加和相乘

![image-20200215093454829](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215093454829.png?raw=true)

这样刻画高阶链接的方式就非常简单了，通过堆叠这样的GAT层就可以得到任意阶的信息。

​	在最终的预测层，通过将GAT各个层的输出连接起来就可以将目标节点周围从1到L阶的所有节点都考虑进来，同样的策略得到用户节点和商品街店的最终表示，通过点积打分得到推荐得分即可。这里同样会产生推荐的一个pair-wise loss

![image-20200215095205467](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215095205467.png?raw=true)

将这个loss和KG embedding的loss综合起来便是整个模型的一个loss

![image-20200215095247251](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215095247251.png?raw=true)

## Experiments

### 	Datasets

​	三个公开的数据集：**Amazon-book**  、**Last-FM** 、**Yelp2018** 均保留至少有10次交互的用户和商品。KG部分使用Freebase并将其实体与商品根据标题对齐。

### 	Baselines

- ​	**FM** 最基础的独立建模方法，使用用户、商品以及相连实体的ID作为输入特征
-    **NFM** 使用NN加强FM的方法
-    **CKE** 使用知识图谱中的嵌入作为矩阵分解的正则化的方法
-    **CFKG** 将用户通过购买关系嵌入到知识图谱中的方法
-    **MCRec** 使用连接用户和商品的meta-path的方法
-    **RippleNet** 综合使用正则化和meta-path的方法，在图上进行传播
-   **GC-MC** 在用户商品交互图上使用GCN的方法

### Results

​	![image-20200215102434480](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20KGAT_%20Knowledge%20Graph%20Attention%20Network%20for%20Recommendation/image-20200215102434480.png?raw=true)

## 个人见解

​	这篇文章的模型本质上就是KG+GAT，将这两个都针对图数据的方法依次使用来增加模型的承载力，其实这样的操作在思想上和实现上的创新性都很有限，也没有让人感觉巧妙的结合点，就是单纯的模型相加，然后对比效果，思路启发上意义不大，最多当作一个应用的实例来学习，同时这种基于图嵌入的方法都不可避免地陷入冷启动问题，这也是其实现中选择筛掉出现次数小于10次的数据的原因，这一点是非常大的局限性，在长距离上的注意力分值的解释性没有那么清晰，分数的大小跟结果的解释性不是那么直接。