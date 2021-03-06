# Learning over Knowledge-Base Embeddings for Recommendation

------

## Motivation

​	在CF类的推荐方法中，除了最原始的实现交互，一些非结构化的数据如：评价、图片等也被用于加强对用户和商品的表征，但是一直没有一个统一的显式关系来表示这些各种不同的信息之间的关系，但是知识图谱的出现使得我们看到了一种异质的实体和关系下的统一表示方法，所以我们尝试使用结构化的知识图谱来对用户和商品进行统一嵌入，再使用一种改进的CF来进行个性化的推荐。

## Model

​	在推荐模型中，需要用到各种不同的实体，比如在一个基本的商品推荐模型中可能存在的实体有：用户、商品、品牌、商品类别、描述词。基于这5种实体，我们希望能够像知识图谱一样将其纳入到一个统一的表述中，所以我们定义了6种关系来链接这5种实体。

- 购买（buy）用户和商品之间的关系。表一个用户购买了一个商品
- 属于某个类别（belong_to_category）商品和类别之间的关系。表一个商品属于一个类别
- 属于某个品牌（belong_to_brand）商品和品牌之间的关系。表一个商品属于一个品牌
- 提到描述词（mention_word）用户或商品和描述词之间的关系。表一个用户或商品的评价中提到了一个描述词
- 共同购买（also_bought）商品和商品之间的关系。表两个商品被一个用户一同购买
- 共同浏览（also_view）商品和商品之间的关系。表两个商品被一个用户一同浏览

使用这5个实体和6种关系，我们就能够得到一个异质的网络图。

![image-20200205102039759](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Learning%20over%20Knowledge-Base%20Embeddings%20for%20Recommendation/image-20200205102039759.png?raw=true)

​	有了这样一个类似知识图谱的网络图之后，我们使用类似TransE的方法将其中的实体和关系统一嵌入到向量空间中。TransE的基本思想是让尾向量与头向量加关系向量接近，在这里我们将这种转换关系表示为trans（）函数，衡量转换后向量距离的函数成为d，则对于所有随机初始化的实体向量和关系向量来说，需要最小化以下的目标函数。

![image-20200205102800630](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Learning%20over%20Knowledge-Base%20Embeddings%20for%20Recommendation/image-20200205102800630.png?raw=true)

​	其中St和Sh分别表示将头和尾进行随机替换后形成的负样本，使用hinge loss来定义损失。在实现中，d函数采用L2距离，而trans函数采用向量相加的形式，即![image-20200205102915328](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Learning%20over%20Knowledge-Base%20Embeddings%20for%20Recommendation/image-20200205102915328.png?raw=true)

​	在模型训练完成之后，对于一个目标用户产生推荐就可以使用统一的向量，对于所有的商品，采用距离![image-20200205103205196](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Learning%20over%20Knowledge-Base%20Embeddings%20for%20Recommendation/image-20200205103205196.png?raw=true)来作为得分进行排序实现推荐。

## Experiments

### Datasets

​	使用常用的Amazon数据集，选择其中的四个板块：CD、Clothing、Cell Phone、Beauty

### Baselines

- ​	**BPR**  贝叶斯个性化排序方法
- ​    **BPR_HFT**  使用BPR改造的HFT方法来进行top-N推荐
-    **VBPR**  使用图片的BPR方法
-    **DeepCoNN**  基于文本评价的推荐方法
-    **CKE** 综合了文本、图片、知识的SOTA知识推荐方法，使用知识图谱作为正则化
-    **JRL** 使用多种模型来共同建模的方法

### Results

![image-20200205105027383](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Learning%20over%20Knowledge-Base%20Embeddings%20for%20Recommendation/image-20200205105027383.png?raw=true)

## 个人见解

​	本文思路是使用知识图谱的方法来解决推荐问题，将用户和商品之间的推荐关系抽象为“购买”关系的预测，通过已有的购买关系以及额外定义的其他辅助数据上面的关系来形成一个大的知识网络，再按照知识图谱的方法进行嵌入，这种思路还是非常有创新性的，也给人一定的启发，比如三元组的概念可以应用在很多问题上，一旦能够抽象为三元组，则能够使用知识图谱的思路来进行异质的嵌入。这个方法是构建知识图谱与其他的使用知识图谱的方法不一样，有的方法采用对齐的方式来引入现成知识图谱中的实体，而这个方法则是整个训练一个知识图谱。当然这种方法也存在一些问题，比如仅仅的购买关系是不是太单薄，以及冷启动无法缓解的问题。