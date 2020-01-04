# Is a Single Vector Enough? Exploring Node Polysemy for Network Embedding

------

## Motivation

​	在之前的网络嵌入模型中，几乎所有的模型都假设每一个节点都只有一个嵌入，但这在某些场景下是不合适的，因为节点可能扮演者多种不同的角色。如下面的场景：一个用户同时喜欢喜剧电影和恐怖电影，则传统的方法中喜剧电影和恐怖电影被认为是相近的节点而具有相近的嵌入，但这显然是不合理的，其二者并没有本质上的联系。再如下面的例子中，用户在进行购物时可能扮演不同的角色从而产生不同的需求，如果将多种角色合并处理显然会产生一定的局限性和相互的干扰。

![image-20200103100146792](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103100146792.png?raw=true)

​	所以，本文的方法就是假设每一个节点有多个表示，如何在这种前提下设计学习的算法，同时将已有的一些网络表示学习的方法拓展到节点的这种多义表示上。

## Model

​	首先，以最基础的DeepWalk方法作为基础，我们在其基础上进行多义的拓展，之后同样会介绍其对应的PTE和GCN版本。

### Polysemous DeepWalk

​	在传统的DeepWalk中，每一个节点对应一个中心表示Ui和一个上下文表示Hi，分别作为其充当中心节点和上下文节点时在目标函数中的表示，现在我们假设每一个节点具有k个角色，即k个表示，为了简单我们定义所有节点具有相同的k，这里k的大小可以由经验得到，如估计的大致商品分类等。现在每一个用户对应两个矩阵Ui和Hi，每一个矩阵包含k个D维的向量。

​	保持DeepWalk中的随机游走采样和Skip-gram模型不变，因为我们有了k个向量，所以每一次使用语料库基于Skip-gram模型来优化参数时需要指定当前优化的应当是哪一个向量。更一般的讲，原来，对于一条语料o，其各个角色的分布定义为p(o)， 对于每一个节点，其角色分布定义为p(ui)，对于语料o中的节点ui，我们定义其角色分布为p(ui|o) 现在对于每一条语料，我们需要优化r次，每一个让语料o中的每一个节点根据其p(ui|o)采样出一个当前的角色，然后优化此角色对应的那个向量参数，因为直接对变形之后的目标函数进行极大似然估计不方便操作，所以我们取目标函数的下界，其对应的采样思路即为上文中描述的那样。

​	![image-20200103102907583](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103102907583.png?raw=true)

​	目标函数中s（o）代表语料o进行角色采样之后每一个节点所处的角色。

​	现在还需要解决的问题是如何定义我们的角色分布，一个直观的想法时节点的觉得应当与其连接的边数有关，所以我们采用的是对图的邻接矩阵A进行矩阵分解的办法![image-20200103103236357](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103103236357.png?raw=true)将A分解为一个n×k的P向量，其中每一行便是对应节点在每一个角色取得的分值，再进一步将其归一化为概率即可得到每一个节点的角色分布p（ui）![image-20200103103443378](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103103443378.png?raw=true)

​	有了这个之后怎么得到一个语料o的分布呢，我们使用一个直接的办法，将语料o中所有的节点分布进行平均作为语料o的分布![image-20200103103614597](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103103611213.png?raw=true)

​	现在只差最后一个了p（ui|o） ，在语料o中，一个节点所处的角色应当由此语料和本节点的属性共同决定，取二者的最小值是一个情理上说得通的解释![image-20200103103901774](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103103901774.png?raw=true)得到的结果为了保证为概率还需要进行归一化操作。

​	现在我们可以总结一下我们的方法相比传统的DeepWalk方法拓展了哪些东西。首先需要对邻接矩阵进行分解得到角色的先验，其次，在优化过程中原本每一条语料进行一次的优化被拓展为多次，每一次根据上面的角色分布采样得到角色，优化语料中对应角色的参数，主要的计算量增加的地方就是进行r次角色采样，这里的r同样是一个新引入的超参数。

​	训练阶段完成之后如何在下游任务中使用不同的嵌入呢，一种最直接的方式是将其直接连接为一个长度为D×k的向量，但是在实验中我们使用一种更合理的方法，即将每一个向量与其先验概率相乘后再连接。

![image-20200103104712582](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103104712582.png?raw=true)

### Polysemous PTE

​	同样以PTE作为代表，将多义性引入这种异质的网络中。

​	PTE中的多种节点可以分割为多个二分图来进行刻画，所以下面再二分图的前提下来应用多义性改造，对于一个二分图，使用同样的随机游走可以得到语料库，但这里我们限定取样时的窗口大小为1，即只使用集合A中的节点来预测集合B中的节点，那么原本每个节点的Ui和Hi两个表示就可以变为集合A表示为Ui，集合B表示为Hi，其余的目标函数与同质图上的DeepWalk方法均没有区别，另外还需要注意，因为我们的图变为了二分图，所以相应的矩阵分解也应当进行改变。

![image-20200103110556276](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103110556276.png?raw=true)

### Polysemous GCN

​	在GCN中，每一层是将邻居节点上一层的信息进行聚合后再线性映射，同样的改造思路，每个节点再每一层应当只在相同角色之间进行聚合。

![image-20200103110836758](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103110836758.png?raw=true)

​	那么问题就变成了如何找到相同角色的邻居呢，因为我们在矩阵分解之后得到了每一个节点在每一个角色下的分值，则对于两个节点来说，使用其对应的k维角色向量进行交互，就可以得到两个节点在哪个角色下存在连接，进而找到分角色的邻居集合。



## Experiments

### datasets

- ​	**BlogCatalog** 一个社交网络数据集，使用博主之间的链接关系来预测博主的兴趣分类
- ​    **Flickr**  一个多媒体网站的用户之间的社交关系数据集，预测用户所加入的小组
- ​    **MovieLens**  一个常用的推荐系统的数据集，作为二分图来预测用户和商品的连接关系
- ​    **Pinterest**  一个图片推荐的数据集，同样抽象为连接预测任务

### baselines

- ​	**DeepWalk**，**PTE**，**GCN** 我们的模型又这些常见的模型拓展而来，所以将其作为对比项来验证我们的模型。
- ​    **NMF** 一个经典的矩阵分解模型，我们使用其来估计数据集中包含的角色。
- ​    **MSSG**  一个NLP中的词嵌入方法，我们使用它来进行网络分析。
- ​    **MNE**   一个基于临界矩阵分解的方法来进行节点的嵌入，我们只将其用在节点分类任务中

### Results

![image-20200103152009450](https://github.com/linzihan-backforward/PaperNotes/blob/master/KDD/%5BKDD2019%5D%20Is%20a%20Single%20Vector%20Enough_%20Exploring%20Node%20Polysemy%20for%20Network%20Embedding/image-20200103152009450.png?raw=true)

## 个人见解

​	这篇文章可以说的DeepWalk等Skip-gram方法的正统续作，整体的motivation还是非常清晰简单的，就是把原本的一个embedding扩充为多个，但是一些实现细节上还是处理的非常好，就比如如何定义角色的分布来进行采样上，尤其是对邻接矩阵进行分解的那一步，感觉非常巧妙，给了老的方法一种新的含义，整个的拓展过程在原来模型的基础上经过一定的抽象和近似而来，每一步给人感觉上合情合理又不乏新意，整个算法唯一的小下次可能是对负采样的处理上没有讲述清楚，给人一种差一口气的感觉，整体的通透感少了一点。时隔几年把最基础的图嵌入方法DeepWalk又拿出来重新审视一遍，从最基础的点出发一点一点解决，最终实现这个听起来不难的想法，还是非常有水平的。