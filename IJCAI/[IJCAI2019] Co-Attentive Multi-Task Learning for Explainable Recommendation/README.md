# Co-Attentive Multi-Task Learning for Explainable Recommendation

------

## Motivation

​	在推荐系统中，结果的可解释性也同样是一个很重要的指标。很多的推荐模型都是一个黑盒子，在产生推荐结果的同时我们无法得知为什么这个商品被推荐，而为了提升用户的体验，有一些工作在可解释性方面进行了一些探索。最早的方法是反推式的，使用固定的解释模板对解决进行解释，后来出现了一些基于嵌入的方法，它们从原始数据中选择那些能提升结果指标的数据作为解释，但是这些方法都或多或少存在一些问题，如：将性能作为目标，而解释性作为伴随，针对这样的问题，我们希望能够让模型将可解释性同样作为学习的目标，使用一种多任务学习（Multi-Task Learning）的方法来得到推荐解决的同时生成解释。

## Model

​	在多任务学习中，一种常用的结构式Encoder-Decoder模式，使用Encoder将输入嵌入到低维向量，然后多个Decoder分别对应不同的学习任务，这里我们使用的是类似的结构。

### Encoder

​	encoder部分的功能是将输入（这里是文本的评价）建模到一个数值的vector上，对于每一个review，其又word组成，我们仅仅将其简化为词袋模型，使用每一个词的word embedding之和作为一篇review的embedding，对于用户u的第i个review，其向量表示为dui。

![image-20200130163037132](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130163037132.png?raw=true)

​	同时为了考虑其他的与评价无关的用户特征，我们同样为每一个用户和商品对应一个可查找的embedding hu

### Multi-Pointer Co-Attention Selector

​	在得到所有的review和user、item的embedding之后，需要从中选择出那些重要的、包含用户和商品特性的特征点，在之前的工作中这一步使用的是TextCNN的方式，而这里我们使用一种层次化Attention的方式，第一层Attention选择出一个最有用的review，第二层再从这个review中选择出最有用的一个关键词，将这个关键词作为后面打分和生成文本的依据。

​	首先我们介绍第一层的Attention结构，在之前的工作中也有使用Attention来筛选评价的方法，但是那些方法都是使用的两个独立的self-attention方式，用户侧和商品侧为两个平行的模型进行计算，这样无法获取商品用户之间的交互信息，所以这里我们使用co-attention的方式，对于用户侧的ld个评价和商品侧的ld个评价，我们可以两两计算一个打分，得到一个打分方阵。

![image-20200130164343207](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130164343207.png?raw=true)

​	这个方阵中每一行对应用户的一个评价和商品的所有评价之间的关系打分，每一列同理，所以我们从行列两个方向进行pooling就可以从所有的评价中选出重要的评价。

![image-20200130164702056](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130164702056.png?raw=true)

​	这样得到的a向量和b向量中每一个值都对应一个评价的重要性得分，与之前工作中加权求和的方式不同的是，这里我们需要从ld个评价中选出最重要的那一个，而不是将这ld个综合考虑，所以这里我们就需要用到argmax函数，但是这个函数是不可导的，没有办法端到端进行训练，所以我们需要找一个近似的替代方案。

![image-20200130165047857](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130165047857.png?raw=true)

​	这是一种叫做Straight-Through Gumbel-Softmax 的函数，其中gi是可以改变的噪音，改变这个噪音会使得函数的最大项产生改变，我们在前向传播中在qi上使用argmax，而在反向传播中短路这个argmax，直接使用qi的梯度进行近似，这样就可以一定程度上近似。将这种方法得到的one-hot向量称作Gumbel（a）则第一层得到的表示可以通过矩阵乘法得到。

![image-20200130170431208](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130170431208.png?raw=true)

​	至此，在评价级别的attention就完成了，但是这还是不够的，还需要更细粒度的关键词级别的第二层attention。在上一步中我们可以得到一个最有用的评价du‘ ，在这个评价中我们使用一个关键词提取工具Microsoft Concept Graph（https://concept.research.microsoft.com）来得到关键词序列Cu，这样其实我们就跟上面第一层的attention方法进行了统一，上面输入的是review序列，这里输入的是词序列，计算权重的方法都相同。

![image-20200130171646905](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130171646905.png?raw=true)

​	但是需要注意的是，经过实验测试，发现使用mean-pooling方式效果最好，所以这里不使用max-pooling

![image-20200130171805971](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130171805971.png?raw=true)

​	之后就是完全相同的softmax近似部分，得到的结果是一个最重要的词，我们把两个阶段的attention结果综合考虑，将最重要的评价和最重要的关键词进行拼接![image-20200130171941294](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130171941294.png?raw=true)至此，就得到了这个selector的输出，单一的输出显然表现力还不够，所以我们将上面的步骤重复多次，选择不同的噪声输入softmax公式中，就可以得到多组结果，将多组结果综合，得到最终的用户表示和商品表示。

![image-20200130172126353](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130172126353.png?raw=true)

### Multi-Task Decoder

​	在得到用户和商品的表示之后，Decoder就分为两个，分别进行打分和生成解释两个任务。

#### Rating Prediction

​	打分采用的方式就是最传统的FM的方式，使用两个vector来考虑一阶和二阶的交互，而Loss使用的就是L2距离。

![image-20200130173036874](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130173036874.png?raw=true)

![image-20200130173048488](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130173048488.png?raw=true)

#### Explanation Generation

​	在评价生成任务中，因为我们需要生成一个新的句子而不是从原始数据中检索一个已有的句子，所以这里实际上结合了一些文本生成的方法。在文本生成中，模型都是自回归的，所以GRU就是一个很好的方法。在GRU中需要一个初始状态作为起始，考虑用户和商品两方面，我们将两个Vector和评分计算到一个公式中

![image-20200130173421602](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130173421602.png?raw=true)

这里使用的是向量化的评分r，即将评分数值离散化到整数再one-hot。有了S0之后按照GRU的方法依次生成每一个词即可。

![image-20200130173604458](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130173604458.png?raw=true)

![image-20200130173610715](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130173610715.png?raw=true)

这个输出ot就是整个词典上的分布，一个|V|维度的向量，选择最大值作为输出词即可。

关于Loss的制定，有两种方法，即我们希望得到的文本既要包含那些关键词又要尽可能地贴近原label中的文本，所以两个loss分别来最大化两个log loss

![image-20200130191407462](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130191407462.png?raw=true)

​	这个loss的含义是：对于每一个生成的word，将其分布中那些关键词对应的最大值取出来，希望这些值最大化。

![image-20200130191536847](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130191536847.png?raw=true)

​	这个loss形式上就比较简单了：就是希望最大化每一个词所对应的label的概率值。

​	至此，整个模型从输入到Loss所有计算都是可导的，可以进行端到端的训练，总共有三个目标函数，将其相加作为多任务学习的Loss

![image-20200130191718902](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200130191718902.png?raw=true)

## Experiments

### Datasets

​	使用的三个数据集均为常用的Amazon数据集中的三个部分：Electronics、Movies&TV、Yelp

### Baselines

​	检索式的解释性方法：

- ​	**Lexrank** ：一个基于随机图计算文本相关重要性的方法
- ​    **NARRE**、**RLRec** ：这两个都是使用Attention方法选取重要句子或评价的方法

   生成式的解释性方法：

 **NRT** ：基于评分和评价中的词分布使用RNN来生成解释的方法

传统的推荐方法：

- **PMF**
- **NMF**
- **SVD++**

### Results

![image-20200131094807659](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200131094807659.png?raw=true)

![image-20200131094818513](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Co-Attentive%20Multi-Task%20Learning%20for%20Explainable%20Recommendation/image-20200131094818513.png?raw=true)

## 个人见解

​	本文与其说是可解释性推荐系统，不如说是推荐系统+文本生成，推荐系统生成评分，文本生成生成理由，然后两个任务放到一起训练，只能说这两者结合的方式有点新意。整个结果中最复杂的应该就是selector部分，将评价中的关键词选出来，FM在关键词的基础上打分，GRU在关键词基础上生成单词。总体上看好像有点意思，但是细想道理上又有点说不通，比如两个词之间的文本相似度跟喜好契合度似乎不是一个概念，使用的各个评价中独立的词而没有不同评价之间相似词的关系，相当于每一个评价割裂考虑。具体细节上也有一些耐人寻味的地方。虽然细节上有点不透，但是大体的框架上还是提供了一种新的领域结合的思路，也启发我们推荐系统的可解释性其实可以借鉴NLP中的各种子领域。