# Matching User with Item Set：Collaborative Bundle Recommendation with Deep Attention Network

------

## Motivation

​	在现有的推荐模型中，大多数针对的都是单个物品的推荐，但是在有一些现实场景中多个物品需要捆绑在一起共同推荐，针对这样的需求，我们设计了一个基于DAN的神经网络模型来对用户和商品集合进行建模，并使用多任务学习的方法丰富数据，最终实现多个商品共同推荐的目标。

## Model

​	我们的目标是将一捆商品作为一个整体推荐给用户，那么直接将一捆作为一个独立的实体采用原来的方法不就可以了吗，显然，这么做会带来很多新的问题，比如捆的定义可变动性很大，随便一种组合就能产生一个新的商品集合，直接将捆作为个体会导致无法处理冷启动问题，其次，用户与捆之间的交互相比与商品的交互来说稀疏了非常多，如此稀疏的数据非常不利于直接地建模。所以，我们的模型不为捆学习独立的表示，而是将捆作为其中商品的一种聚合方式，考虑到不同商品在集合中地位不同，我们采用Attention的方法生成用户独立的权重。

![image-20200305142731445](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305142731445.png?raw=true)

​	其中bs为第s捆商品的表示，vj为其中单个商品的表示，α（i，j）代表用户i对商品j的打分，具体地实现为两层的MLP

![image-20200305143043527](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305143043527.png?raw=true)

​	但是考虑到这样的MLP无法学习到用户和商品之间的低阶关系，所以我们简化其为直接的向量点乘，但是为了将商品在捆中的表示区分开，我们为每一个商品再赋予一个全新的商品embedding专门用在这里

![image-20200305143516143](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305143516143.png?raw=true)

​	在得到了捆的表示之后，使用用户表示、商品表示、捆表示，我们便能够进行商品打分和捆打分的任务，为了保证两个任务之间相互促进，我们将两个任务的底层模型固定为同一个

![image-20200305144126867](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305144126867.png?raw=true)

​	最后两个任务的目标函数完全相同，都采用BPR Loss

![image-20200305144353228](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305144353228.png?raw=true)

![image-20200305144400101](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305144400101.png?raw=true)

## Experiments

### Datasets

- Netease Cloud Music 网易云音乐数据集，单首歌作为商品，一个合集作为捆
- **Youshu** 有书数据集

### Baselines

- **BPR** 
- **NCF** 
- **BR** 一个两步的模型，首先学习用户和商品的表示，再直接聚合商品作为捆
- **EFM** 考虑了统一捆中的商品共现关系，分解用户-商品-捆交互和商品-商品-捆交互

### Results

![image-20200305145511412](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Matching%20User%20with%20Item%20Set_%20Collaborative%20Bundle%20Recommendation%20with%20Deep%20Attention%20Network/image-20200305145511412.png?raw=true)

## 个人见解

​	这篇文章的模型看上去是极其的简单，让人不敢相信这是IJCAI2019的文章，整个模型就一个Attention+几个MLP就完事了，从实验结果来看，提升也仅仅在百分之几，似乎并不大。但是这篇文章新就新在问题和数据集上，推荐任务大家翻来覆去使用的还是那几个开源的数据集，而本文使用新的数据集配合较新的集合推荐的任务，一下子突破了常规的用户-商品二元关系，其作为最简单的开创工作后面一定还有更复杂的模型等着被接受，正如文章最后所说，商品之间的共现和用户兴趣的时序性都还没有考虑，而这两个问题在传统推荐模型中已经研究很多，把那些方法拿过来融进模型是不是就又生成一些新文章呢，20年值得期待。