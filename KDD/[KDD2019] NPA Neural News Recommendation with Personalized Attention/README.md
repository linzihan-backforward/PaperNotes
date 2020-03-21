# NPA Neural News Recommendation with Personalized Attention

------

## Motivation

​	在新闻推荐中，如何刻画新闻的表示和用户的表示是一个关键点，很多工作一般把这两步骤作为前后级，利用得到的新闻表示建模用户表示，但是在实际中一个用户可能有多方面的兴趣，不同用户又可能有相同兴趣，导致用户因为不同的原因喜欢一个新闻，所以本文将用户的ID引入到新闻表示层面，来进行个性化的新闻表示的学习，同时利用词级别和新闻级别两层的Attention构建用户表示。

## Model

​	与其他的推荐方法一样，我们的模型分为三个部分：新闻编码器、用户编码器、点击预测模块，三个部分是前后相连关系。

### News Encoder

​	首先，每一个新闻的标题表示为词向量矩阵，然后在矩阵上进行TextCNN的操作来得到每一个词包含上下文的表示，之后传统的TextCNN会进行pooling操作来得到句子的表示，而我们使用个性化的Attention替换。具体的使用用户ID的embedding作为query，所有词的上下文表示作为Key，得到每一个词对应的权重作为输出，将所有词进行加权聚合，与之前工作最大的不同点在于我们的query不再固定，而是与用户相关，能够形成词级别的解释性。

![image-20200316145012594](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20NPA%20Neural%20News%20Recommendation%20with%20Personalized%20Attention/image-20200316145012594.png?raw=true)

### User Encoder

​	得到用户的表示与新闻的表示在方法上是一样的，只不过输入变为了所有上一步输出的新闻表示，进行一样的以用户embedding为query的attention之后，聚合得到用户的表示

### Predictor

​	与其他的使用二元打分函数直接得到用户与目标新闻的CTR不同，这里我们同时计算一个用户与多个新闻之间的关系，每一组共有K+1个新闻，其中只有一个是正样本，需要最大化正样本在所有K+1个样本中的相对得分

![image-20200316150106297](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20NPA%20Neural%20News%20Recommendation%20with%20Personalized%20Attention/image-20200316150106297.png?raw=true)

​	整体的模型就是两层的attention结构，非常容易理解

![image-20200316150215051](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20NPA%20Neural%20News%20Recommendation%20with%20Personalized%20Attention/image-20200316150215051.png?raw=true)

## Experiments

### Datasets

​	来自于MSN News 的真实数据集，包含一个月的用户浏览记录

### Baselines

- **LibFM**

- **CNN**

- **DSSM**

- **Wide&deep**

- **DeepFM**

- **DFM**

- **DKN**

  

### Results

![image-20200317155102963](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20NPA%20Neural%20News%20Recommendation%20with%20Personalized%20Attention/image-20200317155102963.png?raw=true)

## 个人见解

​	说实话这篇文章我认为非常水，但是作为新闻推荐的related work不得不通读一下，整篇工作的核心idea就是赋予attention中无意义的query参数以用户id的含义，形成用户的可解释性权重。作为19年kdd的工作如此通俗的idea和简单的模型能够中与其他的非常solid的工作形成了鲜明的对比，虽然哪里都make sense，也没有明显的缺陷，但是就是让人觉得是大水文哈哈。