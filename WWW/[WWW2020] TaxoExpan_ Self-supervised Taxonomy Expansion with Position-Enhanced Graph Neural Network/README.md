# TaxoExpan ：Self-supervised Taxonomy Expansion with Position-Enhanced Graph Neural Network

------

## Motivation

​	分类信息是指一系列实体或概念由包含关系所形成的树形结构，研究如何构建便是分类学，得到的信息也被广泛的用于搜索引擎和在线电商中，但是由于网络内容的飞速发展，有很多新的名词或者事务不断地出现，老的分类结构中没有这些新的信息，所以也就无法处理这些大量的知识，所以如何动态的拓展生成分类信息是一个值得研究地方，利用已有的结构采取自监督的方法生成新的分类信息便是本文中解决的问题。

## Model

​	首先，一个现成的分类信息可以表示为一个有向无环图的结构，每一个实体或概念有其唯一的父亲。而拓展工作则是将一些额外的新的实体或概念加入到已有的图结构中，在本文的模型中，我们假设不对已有图结构进行修改而只进行添加，一个分类信息拓展的例子如图所示：

![image-20200403163439501](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403163439501.png?raw=true)

​	如果我们将整个图上的实体看作是独立的话，那么生成一个分类图的概率可以看作是每一个节点在其父节点之下的概率，所以整个网络拓展工作从概率角度可以看作是最大化新的有向无环图的概率

![image-20200403164425998](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403164425998.png?raw=true)

​	直接对上式进行求解的话，一是无法穷举可能的父节点结构，二是无法保证不改变已有的网络结构，所以我们简化这个问题，改为寻找新的实体的父节点并最大化相应的对数概率

![image-20200403164810023](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403164810023.png?raw=true)

​	经过这样的简化，拓展问题可以看作一个一个加入新实体，为其寻找父节点的过程。

​	模型需要建模得关键就是实体节点和其父亲的关系，将当前的实体节点叫做查询，那么其真正的父亲可以叫做答案，下面的主题模型便是围绕这二者展开的，首先对于查询，因为其表示一个新的实体，所以需要使用其名称或者额外的文本信息来得到其向量表示，而对于答案，因为其存在于一个已有的分类结构中，所以其向量表示应当不仅仅包含其名称的文本，而应当包含结构中的邻居结构来帮助更好的进行预测。所以我们定义一个答案节点应该学习到其局部连接信息，具体一点，应包括其本身、父节点和孩子节点三代

![image-20200403170502676](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403170502676.png?raw=true)

​	这样对于每一个答案节点都有一个小的图结构，就可以使用GNN的方法来建模。具体地，可以将这样的图结构使用原始的GCN或者GAT得到每一个节点的特征向量，之后将这三代节点表示进行平均或者求和来得到这个小子图的向量表示。但是使用原始的GAT无法对这三代节点的不同地位进行区分，其我们还需要建模相对于询问节点的相对位置关系，所以我们改造GNN，得到一种位置加强的GNN来对图信息进行建模。

​	其实非常简单，我们为每一个节点再额外设置一个位置向量p，就像BERT中的那样，然后将其与原始输入连接之后，再参与后续每一层的信息传递，这样就能将相对位置信息传递给邻居节点。

![image-20200403172342689](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403172342689.png?raw=true)

​	在考虑了相对位置之后，图上的平均也可以变为加权平均，不同的位置赋予不同的权重，或者将处于同一个位置的节点进行平均，不同位置结构连接。

![image-20200403172512614](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403172512614.png?raw=true)

![image-20200403172521185](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403172521185.png?raw=true)

​	在得到的查询的向量表示和答案的向量表示之后，二者的匹配程度就非常简单的了，可以使用双线性打分函数或者连接之后通过MLP，都是可以的。

​	在整体的模型介绍完之后，因为我们是自监督的模型，所以训练数据来自于已有的分类结构，对于一条存在的边，我们类比于知识图谱补全那样采样N个对应的负样本来共同形成数据集，这样我们的数据集由一系列二元组集合构成，每一个集合包含一个正样本和N个负样本。然后我们使用一个多分类的Loss来更好的利用负样本，而非二分类目标

![image-20200403183812027](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403183812027.png?raw=true)

整体模型结构图如下：

![image-20200403184436121](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200403184436121.png?raw=true)

## Experiments

### Datasets

- 来自 FoS的一个分类图，包含660k个实体和至少700k个关系
- 由上面的整体图导出的仅包含计算机科学下实体的子图
- SemEval Benchmark dataset

### Baselines

- **Closest-Parent** 余弦距离最近的点作为答案
- **Closest-Neighbor** 距离加上于其孩子最近的点作为答案
- **dist-XGBoost** 将手工选出的特征输入xgboost预测得分
- **ParentMLP**  将查询和答案拼接送入MLP
- **DeepSetMLP** 将答案的孩子节点加入到答案的特征中

### Results

![image-20200404103221961](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2020%5D%20TaxoExpan_%20Self-supervised%20Taxonomy%20Expansion%20with%20Position-Enhanced%20Graph%20Neural%20Network/image-20200404103221961.png?raw=true)

## 个人见解

​	本文是一个比较有趣的GNN的应用模型，分类图谱拓展。一个分类图谱是一个有向无环图，任务是将新的节点加入到图中的合适位置。由于本文将这个问题进行了很大的简化，只考虑新节点独立的关系而不考虑之间可能存在的关系，同时只加入到叶子节点上面。经过这样的简化之后，其实有点像一个简化版本的知识图谱补全问题了，既然其模型的思想是将一个节点周围的网络信息用GNN来得到，那么是不是同样的思路可以用于知识图谱上面呢？本文的应用角度不能说是眼前一亮，但是也有一定的新颖性，主要是任务上面比较少见。从中可以看到任务对工作的重大影响。