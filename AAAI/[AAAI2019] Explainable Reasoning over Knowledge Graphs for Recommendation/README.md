# Explainable Reasoning over Knowledge Graphs for Recommendation

------

## Motivation

​	将知识图谱的信息融合进推荐系统近年来成为了推荐方向的一个研究热点。通过将各种不同的实体和关系加入用户和商品的二部图中，用户和商品之间形成的各种复杂的路径为刻画用户喜好和商品特征提供了额外的丰富信息，但是已有的工作中并没有充分挖掘这种连接关系来建模用户偏好，尤其是其中的序列相关性。本文提出了一种使用图上路径的循环网络模型，能够充分挖掘路径的序列性并在一定程度上提供可解释性。

## Model

​	我们的模型是定义在序列数据上的，在介绍模型之前首先讲一讲需要的序列数据格式。在KG中存在多种的（h，r，t）的关系，每一个对应图中的一条边，而用户和商品的二部图仅仅存在一条表（u，interact，i）所以将这个二部图融合进知识图谱中，我们便能得到一个具有用户实体和商品实体的知识图谱，定义目标路径为起点是用户，终点是商品的一条路径，中间通过多个（h，r，t）关系连接多个实体和关系。如Alice用户与Castle on the Hill商品之间存在的两条路径。

![image-20200228170859586](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228170859586.png?raw=true)

则我们的模型可以形式化定义如下，给定用户u和商品i，以及（u，i）所形成的K条路径，需要得到一个预测值表示（u，i）之间直接交互的概率，代表使用K条间接的路径推理出一条直连路径的合理性。

![image-20200228171141826](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228171141826.png?raw=true)

我们的模型分为三个部分，如下图所示。

![image-20200228171421631](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228171421631.png?raw=true)

### Embedding Layer

​	在这一层中我们需要将这一条路径变为可计算的向量上，首先对路径进行拆分，一条路径按照实体+关系拆分，每一个路径上的实体和其下一步所连接的关系作为一组，最后一个实体采用特殊的End标记补齐，如上图所示，同时为了对不同的实体进行区分，将实体的类别同样嵌入后进行连接。

### LSTM Layer

​	为了突出序列性，我们采用RNN的方式来处理一条路径所产生的实体关系序列，其中LSTM是一种常用的序列模型，其每一步使用三部分的嵌入相连后作为输入x，与上一步的隐状态h共同形成当前的隐状态，最终，得到的隐层即包含了路径的序列性又包含了语义性，我们将其作为当前路径的向量表示。有了路径的表示之后，还需要得到一个概率作为最终的直连得分，同样一个简单的双层MLP就可以满足我们的需求。

![image-20200228180806093](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228180806093.png?raw=true)

### Weighted Pooling Layer

​	在整个用户-商品知识图谱中，一个（u，i）对之间存在的路径可能不止一条，一个最简单的方法即对每一条独立计算，之后再将所有的路径进行平均。但是很多的研究已经发现，不同的路径对于结果的贡献是完全不同的，所以我们使用一种加权求和来进行聚合

![image-20200228181122912](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228181122912.png?raw=true)

上面的函数中通过调整γ值的大小，可是近似不同的聚合方式，如最大值池化、平均值池化，所以更具一般性，也可以使得我们最终的预测结果更加灵活。

​	与先前的工作相同，我们把推荐任务定义为0、1分类任务，所有的正样本为1，而负采样得到的负样本为0，则使用cross entropy的目标函数即可满足。

![image-20200228181723685](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200228181723685.png?raw=true)

## Experiments

### Datasets

​	使用的两个组合分别针对电影推荐和音乐推荐任务

- ​	**MovieLens-1M** 作为推荐数据集与 **IMDb** 知识图谱相对齐
-    **KKBox** 作为音乐领域的数据集包含了足够的额外知识关系

### Baselines

- **MF**
- **NMF**
- **CKE**
- **FMG** 同为图上的路径方法，使用meta-path

### Results

![image-20200229093934429](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Explainable%20Reasoning%20over%20Knowledge%20Graphs%20for%20Recommendation/image-20200229093934429.png?raw=true)

## 个人见解

​	这篇文章的最主要思想就是把复杂的图结构拆解，转换为之前的模型可以处理的格式化序列数据，其实这种思路的方法有很多，采集路径，形成语料、在预料上运行模型。近年来大家很少使用这类图解析方法是因为有能够直接对图数据运行的模型出现了，并且效果还很好，毕竟路径实际上还是对图的连接关系有所损失，这种简单的LSTM序列处理语料的方法能够取得很好的性能也是没想到的，实验部分对比的baseline也是没有啥强对比项，代码也没有开源，所以这种方法我觉得并不是图上的主流，倒是可以作为以后衔接meta-path方法和GNN方法的一个中间baseline。