# Session-based Social Recommendation via Dynamic Graph Attention Networks

------

## Motivation

​	在社交推荐的场景下我们不仅可以获得用户历史的行为还可以获得用户之间的社交关系，所以用户的喜好不仅仅收到历史行为的影响，还受到其朋友的影响。同时用户的兴趣随着时间是变化的，长期兴趣和短期兴趣可能存在一定的区别，用户可能会受到朋友的短期兴趣的影响也可能会受到朋友长期兴趣的影响，这应当取决于用户当前这段时间（session）的行为，所以我们希望能够使用用户的当前行为来动态的衡量其朋友对他的影响，也就是说朋友的影响是随着时间变化的，为了实现这个目的，我们使用了GAT的模型，每一次根据用户行为来学习朋友的作用并考虑其长短期兴趣的不同。

## Model

​	整体的模型是结合了session-based的推荐模型和图卷积模型两个部分，共分为四个模块：个人动态兴趣检测模块、邻居兴趣表示模块、邻居信息聚合模块和推荐模块，下面针对每一个模块的功能进行介绍，再进行整体的端到端训练。

### Dynamic Individual Interests

​	作为session-based的推荐方法，一个最关键的点就是从当前session中提取用户的短期兴趣，我们将用户的历史行为按照时间（以周或月为单位）划分为不同的session，假设用户的短期兴趣只与当前时间节点所在的session中的行为有关，那么通过将这一序列商品通过一个RNN就可以得到包含时序信息的用户短期兴趣，这也是很多方法中对这种session序列的一个常见的处理方法。具体地，我们使用LSTM来实现。

![image-20200121182134401](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200121182134401.png?raw=true)

### Representing Friends’ Interests

​	我们假设用户周围的朋友会对当前用户的兴趣产生影响，而朋友的影响当时又包含短期兴趣影响和长期兴趣影响两种，我们希望对这两种方式分别建模。

​	对于每一个用户的短期兴趣，我们使用其前一个session的行为作为衡量的依据，即当前时间所属的session的前一个完整session，采用的当时依旧为RNN的方式，为了使得模型尽量简单，这里的RNN与上一个模块中采用同样的参数和结构

​	对于用户的长期兴趣，因为其是不随时间变化的，所以可以考虑为一个用户的表示，所以我们仅仅使用一个embedding向量来作为长期表示，并在训练中进行优化。

​	得到长短期偏好之后，将其拼接起来在经过一次非线性的映射便可以得到用户的整体偏好表示。

![image-20200121183148486](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200121183148486.png?raw=true)

### Context-dependent Social Influences

​	在获取到目标用户的表示和周围朋友的表示之后，就可以来建图了，对于一个目标节点u，其有N（u）个邻居分别对应与其直接相连的节点，采用GAT的邻居聚合的思路，将每一个邻居节点和当前节点来计算权重，然后进行加权聚合的方式。

![image-20200122095921220](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200122095921220.png?raw=true)

### Recommendation

​	在经过多层的GAT聚合了邻居的信息之后，我们就可以得到包含社交信息的用户兴趣表示，考虑到用户在当前session的行为仍然是一个主要的因素，所以我们将两部分的表示进行相连再线性映射后得到用户最终的兴趣表示。![image-20200122100246741](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200122100246741.png?raw=true)

最终的目标函数便是非常简单粗暴的针对所有商品的softmax函数。

![image-20200122100357877](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200122100357877.png?raw=true)

## Experiments

### Datasets

​	分别使用了三个网站爬取的数据集：Douban、Delicious、Yelp

### Baselines

​	传统推荐方法，既不使用时间信息又不使用社交信息。

- ​	ItemKNN
- ​	BPR-MF

​	社交推荐方法，考虑了社交信息

- ​	SoReg
- ​	SBPR
- ​	TranSIV

​	session-based推荐方法，考虑了时序信息

- ​	RNN-Session
- ​	NARM

### Results

​	![image-20200122151500903](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Session-based%20Social%20Recommendation%20via%20Dynamic%20Graph%20Attention%20Networks/image-20200122151500903.png?raw=true)

## 个人见解

​	这篇文章可以算是社交推荐和序列推荐的结合方法，如果但从任何一个方便看都没有太突出的地方，但是其将时序的动态性考虑到了用户图中，这应当是其主要的一个创新点，在社交方向其采用的是GAT的思路来为每一个朋友学习权重，在序列方面使用RNN来获取用户初始的兴趣表示，两个步骤形成一个层次结构，这种方法还是有一些意思的，利用其他的模型得到节点的初始表示然后再利用图结构得到高级表示，这种思路我觉得是一个非常好的图和其他结合的方法。我认为这篇文章中模型的一个不好的地方在于中心用户和朋友之间的不对称性，这样使得对于每一个用户要重新建图，所有用户无法形成一个统一的关系图，这完全是没有必要的，完全可以将当前用户的长期兴趣也加进来，增加模型的可理解性和性能。