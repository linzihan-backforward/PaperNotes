# Sequential Recommendation with Relation-Aware Kernelized Attention

------

## Motivation

​	使用Attention机制来处理序列化推荐任务产生了一类方法，其思想都是使用Transformer中的self-attention应用在序列的商品输入上，学习到不同商品的权重，尽管有不同的变形来处理多种类型的数据，但是它们还都是原本的确定性attention模型，即模型的输出即为结果。我们考虑一种结合概率模型的attention变种，为每一个输出值增加一个隐藏空间用来建模上下文关系，从而将一个多元斜正态分布融合进attention模型来更好的学习各个项的权重。

## Model

​	首先，Attention模型第一步离不开Embedding，选择当前用户的最后n个商品作为序列，为每一个商品对应一个item embedding和position embedding，二者都是一个相同维度的向量作为模型参数，将两个embedding直接相加作为模型的输入

![image-20200310101744860](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310101744860.png?raw=true)

​	第二步便是将输入送入self-Attention模块中，在基本的attention中对于每一个元素输出一个值作为权重，这里我们不再将输出直接作为权重，而修改每一个元素的输出值为斜正态分布的参数，根据输出的参数确定一个分布，再从此分布中采样得到每一个元素所对应的权重

![image-20200310102845809](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310102845809.png?raw=true)

​	其中X为序列的输入，embedding之后的矩阵，C为序列中的商品的共现矩阵，使用全部的数据集得到，MSN即为斜正态分布，其共包括三个参数，具体地数学定义可以查阅相关的分布资料。其输出与attention相同，都是每一个元素对应一个隐层表示H，同样我们的attention结构后面使用FFN来增加表示能力，多个层之间可以堆叠计算。

​	假设我们使用了B层的结构堆叠，那么最终的输出便是第B层的第n个隐藏表示用来预测，其与所有商品嵌入之间的点积作为每一个商品的打分

![image-20200310105144237](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310105144237.png?raw=true)

​	在介绍完模型的整体结构之后，最关键的便是如何使用Attention来得到分布的三个参数。

### Location

​	第一个参数是上面公式中的ξ，其作用类似于分布的均值，这个值是一个决定性的参数，其应当与attention中的得分有直接的关系，所以我们将其定义为Key和Query的得分，这一点与基本的Attention中权重得分一致

![image-20200310105922083](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310105922083.png?raw=true)

### Covariance

​	第二个参数便是协方差，Σ，这个参数的作用是表达多元分布中每一个元与其他元之间的关系，为一个n×n的矩阵参数，这里我们用一个复杂一点的核方法来得到这个矩阵，核方法的思想是首先将每一个元素映射到高维空间，再在高维空间中计算二者的交互，对于每一个元素，计算其与第n个元素的方差

![image-20200310111717448](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310111717448.png?raw=true)

​	得到的方差作为核空间中每一个元素伴随的参数，接下来有三种不同的核函数可供选择，第一个单纯使用两个元素的共现次数计算，第二个使用两个元素的嵌入计算，第三个使用元素嵌入和当前用户的嵌入结合计算，详细的公式表达参见原文，将三个核函数线性求和后作为最终的实现

![image-20200310112311128](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310112311128.png?raw=true)

### Shape

​	此参数α的作用是定义一种偏序应当表达最后一个物品与之前所有物品的关系，所以是一个1×n的向量，其包含两部分，一部分直接使用attention结构计算二者的相关性

![image-20200310113157070](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310113157070.png?raw=true)

​	另一部分使用共现矩阵C的第j行和第n列之间的点积来表达

![image-20200310113417030](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310113417030.png?raw=true)

最后二者共同作用得到α参数

![image-20200310113443901](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310113443901.png?raw=true)

最终的Loss又两部分构成，第一部分为cross-entropy的推荐Loss，第二部分为核方法中估计核参数的Loss

## Experiments

### Datasets

五个序列推荐数据集：Amazon（Beauty，Games）、CiteULike、Steam、MovieLens

### Baselines

- POP
- Item-KNN
- BPR-MF
- GRU4REC
- NARM  使用Attention分别建模长距离和短距离依赖
- HCRNN 考虑了序列的兴趣变化，全局、局部，时序的上下文表示GRU建模
- AttRec
- SASRec 使用Transformer结构处理序列信息

### Results

![image-20200310142147242](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Sequential%20Recommendation%20with%20Relation-Aware%20Kernelized%20Attention/image-20200310142147242.png?raw=true)

## 个人见解

​	这篇文章的核心思想就是将物品之间的共现信息融入Attention权重的建模中，思路上有一定的不同寻常，Attention和概率模型这两个前后时代的方法经过一个简单的修改能够相互连接，有一些关公战秦琼的意思，关于概率模型这方面我了解的不是很多，文中的一些详细的公式也不敢说全部都理解到位，但是经过它这个思路一启发，发现attention和多元分布好像有那么点相似，可以说就是一个自学习的多元分布，根据输入学到一种分布，然后再自动采样出n个值作为权重输出，还是有一点意思，既然将attention看作是一种分布模型，是不是与其他的分布模型LDA、PSA等能够结合呢。