# Unifying Knowledge Graph Learning and Recommendation_ Towards a Better Understanding of User Preferences

------

## Motivation

​	在推荐系统中融入知识图谱是一种有前途的加强效果和解释性的方法，已经有相关的工作尝试使用知识图谱中的实体来作为商品的表示，但是之前的工作都没有考虑知识图谱的不完整性，显然现在的知识图谱是不全的，所以在进行推荐的过程中应该考虑这种缺失性，在产生推荐结果的同时对知识图谱进行补全，本文提出的一个统一模型便能够利用用户和商品的关系来对知识图谱进行补全。

## Model

​	在知识图谱相关的TransH方法中，每一个关系都对应一个超平面，使用一个关系相连的两个实体应当先映射到这个超平面上再进行向量计算，这样通过引入超平面这个额外的学习参数，可以解决多对多的关系的问题，即一个实体上可能连接着多种关系，而多种关系之间无法协调嵌入。所以对于一个三元组（eh，et，r）需要最小化下面的函数。

![image-20200201111239946](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201111239946.png?raw=true)

这就是TransH算法的基本思想。受这种思想的启发，我们将推荐系统中的用户购买行为也抽象为用户和商品这两类实体之间的一种关系，即用户+某种喜好=商品。这样的话我们可以预先定义p中喜好，用户的每一个购买行为都是由这p种喜好关系导致的，也就可以理解为购买关系分为了p类，每一类都有一个超平面，原先用户商品的二元组（u，i）变成了（u，rp，i）的三元组，这样就可以和知识图谱补全问题统一起来了。

​	这样做的话所面临的第一个问题就是关系rp的确定，在知识图谱中两个实体之间的关系r是显式给出的，但是在推荐系统中用户和商品之间是哪一种喜好关系可是不知道的，所以首先就要根据u和i得到对应的rp。

​	与主题模型类似，我们可以事先定义共有P个可取的喜好关系，然后对于一个（u，i）对这就是一个分类问题，一种硬性的分类方式就是使用softmax从这P个里面选一个，这样为了进行端到端训练梯度不断，需要用到Straight-Through (ST) Gumbel SoftMax的方法，其采取不同的前向后向传播方法来近似softmax梯度。

![image-20200201112533626](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201112533626.png?raw=true)

其中未归一化的打分Π可以使用一个相似度函数得到，具体可以是一个点积。

![image-20200201112618357](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201112618357.png?raw=true)

与硬性选择相对应还有软性的，考虑到用户选择商品并不是一个单一的喜好，可能是多种喜好综合的结果，所以软性的rp选择应当时对所有P个关系的加权求和。使用attention的方式即可。

![image-20200201112847630](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201112847630.png?raw=true)

​	在得到了用户的喜好关系rp之后，使用与TransH相同的方法来训练用户和商品向量，即p关系应当使得用户和商品向量在p关系的超平面投影之后相接近。

![image-20200201145058812](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201145058812.png?raw=true)

​	其中wp为关系p所对应的投影向量，为所有用户共享。这样我们就可以使用负采样的方法来最大化负样本最小化正样本。

![image-20200201145219516](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201145219516.png?raw=true)

### Joint Learning

​	在介绍了如何将推荐转化为与知识图谱类似的转换式模型之后，我们还需要使用知识图谱来辅助完成两个目标的共同学习。现在假设我们可以在知识图谱中获取到一些和商品相对齐的实体项，我们将这个实体对应的表示e与原本商品的向量i相加和，使得商品获取到知识图谱中的结构信息，同时根据一个预先定义的1对1的对应关系，我们可以将这个实体的一个关系r与当前的喜好关系p所对应，这样我们的p也可以与图谱中的关系r相加，至此我们的关系p和商品i都有了知识图谱中的信息。

![image-20200201152313684](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201152313684.png?raw=true)

​	使用加和之后的p和i作为推荐所优化的目标表示，同时可以对原本知识图谱中的实体表示e进行微调，实现对知识图谱的补全功能。将两部分的目标Loss进行加和得到端到端模型的目标。

![image-20200201152527174](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201152527174.png?raw=true)

整体模型的结构如下图所示；

![image-20200201152550892](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200201152550892.png?raw=true)

## Experiments

### Datasets

​	两个都是关于电影的书籍评分数据集：**MovieLens-1m** 和 **DBbook2014** ，将其中商品和DBPedia知识图谱中的实体相对应。仅仅保留那些能够与商品对应上的三元组作为知识图谱中的内容。

### Baselines

- ​	**FM** 和**BPRMF** 这是最经典的矩阵分解的推荐方法
-    **CFKG** 使用TransE方法将实体对齐的知识图谱融合进CF方法
-    **CKE** 将多种对其实体的嵌入一同考虑的基于知识图谱的方法
-    **CoFM** 使用共享参数来训练FM和TransE的一种推荐方法

对于知识图谱补全的方法，选择的时最经典的TransE、TransH、TransR方法。

### Results

​	![image-20200202095806926](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2019%5D%20Unifying%20Knowledge%20Graph%20Learning%20and%20Recommendation_%20Towards%20a%20Better%20Understanding%20of%20User%20Preferences/image-20200202095806926.png?raw=true)

## 个人见解

​	这篇文章的创新型我认为还是很强的，不在于使用知识图谱的方法，而在于将用户和商品的二元交互关系拓展到了用户-喜好-商品的三元组关系，从而和知识图谱中的三元组有了对齐的作用，相比于之前的仅仅使用现成的知识图谱中的实体嵌入来加强推荐系统的方法，这样的方法更加类似于将推荐系统融入知识图谱，作为一种变形的子图，我觉得是给了推荐系统发展的一种新的思路，后续的方法可以考虑如何在三元组上构建模型，而不是传统的用户-商品两个平行模型。当然可能是开创性工作的原因，我认为一些地方此文章并没有说清，比如在喜好关系求解部分，使用人为定义的关系来强行和知识图谱中存在的关系进行一一对其，这种做法有点缺乏合理性，并且给模型带来了人为上的难点。