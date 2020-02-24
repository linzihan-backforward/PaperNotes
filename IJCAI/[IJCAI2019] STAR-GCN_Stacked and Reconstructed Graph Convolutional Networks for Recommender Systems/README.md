# STAR-GCN：Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems

------

## Motivation

​	在推荐系统场景中，GCN因为在图数据上的优秀性能而获得了各类SOTA结果，但是很多的模型都绕过了冷启动这个问题，而本文通过堆叠和重构GCN的模型来解决冷启动问题，并且直接使用用户和商品的低维嵌入作为输入进行端到端训练以提高对大数据集的适应能力，同时第一次指出在训练中存在的标签泄露问题，在解决上述问题的前提下，我们提出的模型比同类的GCN方法得到更好的性能。

## Model

​	我们的模型主要组件有两个，Encoder负责使用节点的特征向量和连接结构生成节点的嵌入表示，而Decoder则负责根据这个节点嵌入恢复原始的输入特征，这样的话对于每一个Encoder会产生一个与任务相关的Loss，而每一个Decoder则会产生一个重建的Loss，同时两个模块不同的组合方式会形成不同的模型结构。

![image-20200224144904307](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224144904307.png?raw=true)

​	首先，模型第一步就是要解决输入的问题，之前的方法中有的使用节点的one-hot表示作为输入，但是这样就无法推广到大数据集，所以我们采用一个embedding table的方式将每一个节点对应到一个维度更小的连续空间中并随着训练端到端更新这个table。但是，这样的方法依然无法处理冷启动问题，其新的节点表示没有办法得到，受到NLP中的mask思想的启发，我们mask掉输入中的一部分节点，然后让模型自己重建这个embedding以此来赋予模型自动输出embedding的功能，将table隐藏于模型中。具体地，对于每一个训练batch，我们随机选择其中pm%的节点进行mask，对于每一个选到的节点，有pz的概率直接将这个节点对应的embedding设置为全0，1-pz的概率则保持节点的embedding不变。通过这样的的操作，在遇到新节点时就可以直接使用全0输入，让模型重建embedding。对于一些任务中节点带有一些属性，我们可以将属性经过一个MLP之后得到的feature vector与其embedding连接起来，一同作为模型输入。

​	通过为模型增加一个重建的任务，我们不止能够处理未见到过的节点，同时新的任务也作为原本的评分预测任务的一个正则化，使得模型学习能力更强。

​	对于Encoder模块的实现，我们跟随GC-MC模型的思路，使用周围GCN的方法对节点的邻居进行聚合，这里对于R种评分分开计算。

![image-20200224150602391](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224150602391.png?raw=true)

​	对于Decoder模块的实现，我们采用简单的一个双层全连接网络，将h再映射回输入x

![image-20200224150730807](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224150730807.png?raw=true)

​	因为我们的Encoder-Decoder可以多层堆叠，假设使用了L层，则最终的Loss表示如下

![image-20200224151008088](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224151008088.png?raw=true)

​	其中Lt未有监督的Loss，即我们的评分预测任务，而Lr为无监督的重建Loss，定义如下

![image-20200224151131223](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224151131223.png?raw=true)

​	在模型训练过程中因为每一条边的分数会作为输入数据，而同时要预测的正是这个分数，所以会出现标签泄露的问题，为了解决这个问题，我们在训练中对于每一个商品-用户对，在图中移除它们之间的连接之后，再将各自的邻居作为模型输入，这样避免了过拟合问题。

## Experiments

### Datasets

​	**Flixster**、**Douban** 分别使用用户和商品各自的邻接向量作为节点的初始特征

​	**MovieLens-100K** 、**MovieLens-1M** 、**MovieLens-10M** 使用年龄、姓名、职业、电影名、年份、类别作为初始特征

### Results

![image-20200224153718621](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20STAR-GCN_Stacked%20and%20Reconstructed%20Graph%20Convolutional%20Networks%20for%20Recommender%20Systems/image-20200224153718621.png?raw=true)

## 个人见解

​	本文算是一个短文，整体上比较简略，同时在模型上也基本上是重用的已有模型或者已有结果，但是推荐的文章能种IJCAI肯定有其中的道理，我觉得其出彩的地方在于mask的思想，通过引入NLP中常用的mask思想来解决冷启动问题，这一点真的是太让人惊叹了，原来这种东西还能这么用，让人恨不得拍自己脑袋大呼为什么没想到，同时本文的方法不是一个模型，而是一类模型，通过修改堆叠方式和具体的模块实现方法，可以针对不同的任务目标设计不同的模型，应该算是在解决冷启动问题上各类推荐任务的一个统一框架了。把输入交给模型来重建，这种思想是一个很有趣的点，之后应该还有其他的应用。