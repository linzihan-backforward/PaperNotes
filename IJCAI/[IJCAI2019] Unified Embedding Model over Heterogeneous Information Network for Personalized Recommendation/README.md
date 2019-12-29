# Unified Embedding Model over Heterogeneous Information Network for Personalized Recommendation

------

## Motivation

​	使用异构信息网络来建模用户和商品之间的关系已经被用在了推荐系统上面，相比较于传统的直接使用用户商品交互矩阵的方法，用图得到结构可以找到更多的间接存在联系的用户的商品，从而得到更加一般的表示和更好的效果，处理异构信息网络（HIN）的常用方法便是使用meta-path采样，然后再处理得到的语料库，但已有的方法都是单独处理每一条语料的，即不同的meta-path所形成的不同序列之间的关系没有被找到，这样一些噪声序列会对模型产生干扰，为了解决这个问题，本文提出的模型能够针对每一个商品和用户聚合语料库中多个meta-path的语料，并学习到同一向量空间的用户表示、商品表示和meta-path表示，最终得到更好的模型效果。

## Model

​	在商品推荐的场景中我们关心的是用户和商品之间的关系，对应到图上就是一条用户到商品或者反过来的路径，所以问题可以抽象为从图中已经存在路径来抽取特征尝试预测未存在的路径。所以这里采用的meta-path均为起点为User终点为Item的序列。那么对于每一个meta-path都可以得到一个邻接矩阵C，其中每个元素C(i,j) 定义为用户i到商品j在meta-path的定义之下存在的路径数，这个C矩阵与传统的共线矩阵地位相似，所以首先要补全C矩阵，然后使用多个meta-path的C矩阵来共同预测最终的R矩阵，这里R矩阵定义为用户和商品的隐式反馈，即R(i,j)为1代表用户i点击了商品j，否则为0，模型输出的R~应当与获取到的稀疏R尽量相似。

​	整个的任务流程如下图。

![image-20191228093508560](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228093508560.png?raw=true)

​	好了，下面一步一步详细讲图中的模块(b)和模块(c)

### HIN Embedding

​	上面说到对于每一个meta-path，都可以得到一个(U,I)的邻接矩阵，首先要对其缺失值进行填补，即进行矩阵分解操作，在之前的工作中都是对不同用户商品对的不同C矩阵进行分别处理，相当于丢失了用户层集的关系而只在语料库级别进行操作，这里我们同时考虑用户商品对(i,j)所形成的所有meta-path邻接矩阵C，假设用户和商品在不同的矩阵中具有相同的隐藏因子，即对所有的C进行统一矩阵分解，为了区分不同的C，我们假设一条meta-path同样存在一个隐藏因子(embedding)，基于这样的假设，对于第p个C矩阵中的用户u商品i的预测值如下：![image-20191228094834934](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228094834934.png?raw=true)

其中eU为用户embedding矩阵，eI为商品embedding矩阵，eP为meta-path embedding矩阵，d为共同的embedding维度。然后为了更好的训练，在矩阵分解这一步定义一个训练Loss使得中间这个模块可以进行单独训练而不依赖下一个推荐模块，因为我们最终的预测任务是预测二值矩阵，所以为了简化模型，也定义这里C矩阵填补的目标为二分类，采用Cross Entropy作为Loss，如下图。

![image-20191228100535484](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228100535484.png?raw=true)

最后，因为所有的meta-path在这里是一同考虑的，但他们可能地位不一定相同，所以对不同的p加和时再为不同的meta-path定义一个统一的权值w，这个w怎么得到在下一模块讲到。

![image-20191228100741310](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228100741310.png?raw=true)

至此，(b)部分就完成了。

### Personalized Preferences

​	得到填补之后的C~矩阵后，便是要预测最终的R，即对于不同的Cp~进行聚合，一个最直接的办法显然是对所有的C~直接平均得到，但是上面说了，不同的C~的贡献应当是不一样的，进而，对于不同的(U,I)对，其C~的贡献也是不一样的，但是在实际训练中，如果把每一个用户u中每一个meta-path的贡献p定义为参数的话，会存在一些参数因为数据不足无法训练好的情况，所以现实中我们同时采用上面说的两种贡献值，即分用户的和不分用户的。

![image-20191228101756566](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228101756566.png?raw=true)

其中wG为归一化的待学习的参数，wu,p与eU和eP有关，为了把他们的embedding放到可以直接点乘的空间，需要再分别进行线性映射至同一空间。

![image-20191228102330227](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228102330227.png?raw=true)

最终，模块(c)的Loss也定义为二分类的方式。

![image-20191228102433761](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228102433761.png?raw=true)

然后，非常巧妙地一点，还记得在上一个模块中使用到的常数w吗，就可以直接用这里的wG，需要注意其反向传播梯度只由(c)的Loss决定哦。

### overview

<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228102719795.png?raw=true" style="zoom:67%;" />

​	两个模块分别定义两个Loss Function，模块(b)学习不同的embedding，模块(c)学习各个权重值，两个模块实际上是紧耦合在一起的，克服了其他模型embedding和打分相互割裂所导致的偏差，最终的Loss也是把两个模块的Loss直接相加得到。

## Experiments

### Datasets

​	共使用了三个数据集：Yelp、MovieLens、Douban，每一个数据集共有6种不同的连边方式。

### Baselines

- ​	**MF**  最简单的矩阵分解方法，直接分解评分矩阵

-  **BPR** 贝叶斯个性化推荐，为用户构建贝叶斯模型，再估计参数

-  **FM** 因子分解机 ，直接建模输入的特征之间的二阶关系

-  **HeteRec** 对每一个meta-path矩阵，使用NMF先分解再聚合的方法

- **FMGrank**  对HIN使用MF进行提取，再用FM做推荐   

### Results

<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Unified%20Embedding%20Model%20over%20Heterogeneous%20Information%20Network%20for%20Personalized%20Recommendation/image-20191228143931479.png?raw=true" style="zoom:50%;" />

## 个人见解

​	这篇使用异构网络来进行商品推荐的文章我读完还是有眼前一亮的感觉的，创新性和趣味性都比较高，跳出了之前模型Skip-gram训练方式的束缚，将整个端到端的过程压缩成一个紧密结合的整体，负采样什么的根本不需要，两个Loss一个直接优化目标打分，一个优化网络结构信息，把meta-path这棵老树开出了新花。模型上唯一我觉得有点问题的点是最后加权求和获得最终输出r的时候，两个权值直接相加不就超出1这个权重上线了吗，难道不应该相加后再进行一次归一化吗？