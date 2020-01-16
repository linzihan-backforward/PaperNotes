# Relation Structure-Aware Heterogeneous Information Network Embedding

------

## Motivation

​	对于异质图表示的研究一直是一个热点也是一个难点，异质图中包含的多种节点之间可能具有各种不同的关系，而之前的工作中将那种关系使用一个同一的模型来处理，即对于各类关系均使得其相应的节点的欧几里得空间距离尽量相近，而这会降低不同关系之间的复杂性，所以我们希望使用不同的模型来处理不同类型的关系，并将其统一到一个训练框架之下，为此，需要首先从图中发现不同类型的关系，我们在进行了大量的分析之后发现了两种异质图中的常见关系：ARs、IRs。AR代表包含关系，即两个节点地位不对等，通常是一个大的概念与一个小的概念之间形成的包含关系，如论文-会议关系，作者-会议关系。IRs表示对等关系，即两个节点并没有明显的大小之分，通常是两个不同概念形成的关系，如作者-论文关系。

## Model

### Affiliation Relations and Interaction Relations

​	为了区分异质图中的各种不同的关系，我们首先需要找到区分的指标来对各种关系定量的判断，对于一个图中的关系，其可能直接由一条边来代表，也可能由一条metapath来代表，如在DBLP学术网络中，一个存在的关系为AP，代表作者写了一篇论文，由图中的一条边组成，而APC关系，代表作者写了一篇论文发表在了一个会议上，由一个metapath构成，同理图中还能找到很多这种关系，那么这种关系之间是等价的吗？在以前的工作中它们是等价的，但是现在我们认为它们应当不等价，不同类型的关系应当采用不同的模型来建模，用什么来给关系分类呢？度是一个很关键的因素。对于一个关系（如：APC、APT等）使用其起始和终止节点类型度来定义关系的衡量函数D

![image-20200115190211642](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200115190211642.png?raw=true)

​	其中tu代表起始节点的类型（如A），tv代表终止节点的类型（如C），d代表此类型节点的平均度，这样我们求出来的D（r）是一个大于1的值，且两类节点差异越大，其值也越大。这样我们可以使用此值将关系分为两类，D（r）非常大时为从属关系（AR），D（r）较小时为对等关系（IR），对等关系中节点之间应当具有更高的相似性。

​	为了更清晰地进行区分，我们再设计一个S（r）作为辅助对两种关系进行区分。

![image-20200115191048785](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200115191048785.png?raw=true)

​	其中，Nr表示r关系再图中具有的实例，Ntu表示tu类型的节点再图中的个数，Ntv同理。S（r）同样可以作为一个区分AR和IR关系的指标，利用这两个指标，我们统计了三个实验中使用的数据集，并把其中各种关系进行了分类。

![image-20200115191355297](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200115191355297.png?raw=true)

​	这样我们就可以通过计算D（r）和S（r）来完成关系的分类，两个类别的关系分别使用下面介绍的不同的模型来更好的挖掘其中的信息。

### Relation Structure-Aware HIN Embedding

​	经过上面的定量分析，我们已经能够将图中的关系分为AR和IR两类，为了分别针对两类不同的特征，我们分别设计两个模型来进行学习。对于AR关系中的两个节点，我们直接使用欧氏距离来作为衡量指标，通过优化欧氏距离来使得两个节点尽量靠近，因为AR代表从属的关系，所以理论上其两个节点应当具有一定的相似性。而对于IR关系，其两个节点为对等的交互关系，所以不应当是他们在空间中相靠近，而是采用一个额外的转换操作来作为两个节点的桥梁。

​	对于AR关系，我们希望最小化两点的欧氏距离，所以其公式表示如下：![image-20200116092820512](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200116092820512.png?raw=true)

其中X表示节点的表示向量，w为此关系的权重，有了这个目标之后，我们就可以使用负采样以及HingeLoss来定义针对AR关系的最终目标函数，如下：

![image-20200116092945493](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200116092945493.png?raw=true)

​	对于IR类型，受知识图谱的启发，我们在两个节点之间定义一个转换关系，然后将三元组的距离作为目标。![image-20200116093156540](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200116093156540.png?raw=true)

​	其中Yr表示关系r所代表的转换关系，与AR一样，负采样可以得到最终的目标函数。

![image-20200116093255855](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200116093255855.png?raw=true)

将两种关系的Loss相加可以得到整个模型的目标函数。

## Experiments

### Datasets

​	数据集采用的就是上面提到过的三个数据集以及表中分别列出的关系以及分类。

### Baselines

​	**DeepWalk** 、**LINE** 、**PTE** 、**Esim** 、**HIN2Vec** 、**Metapath2vec** 

### Results

​	本模型得到的低维表示被分别用在下周的节点聚类，连接预测、节点多分类任务上，均取得了SOTA成绩，只展示连接预测的结果。

![image-20200116095102453](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20Relation%20Structure-Aware%20Heterogeneous%20Information%20Network%20Embedding/image-20200116095102453.png?raw=true)

## 个人见解

​	这篇文章思路山和质量上我认为还是非常高的，其主要的点在于将异质图中不同类型的关系进行了区分，同时引入了类似TansE的建模方式到embedding的学习目标之中，角度还是有一些新奇的，在延续性上应该算是metapath-based方法的又一个延申，将原本采样之后进行的统一模型又分为了两个模型。但是我认为这种方法相比于基础的HIN方法添加了很多的限制，也有了一定的局限性，比如在关系的划分上，并没有一个明显的界限用于所有的数据集，只能针对每一个数据集的统计特征来设定个性化的划分界限，对于一些复杂的规模更大的数据集，进行这种特征统计也是相当耗时的，同时此模型并没有考虑节点的附加属性，这也可能是一个未来的工作，即在属性图上设计针对不同关系的不同模型。在GCN中也有一些变形是能够处理HIN的，其理论上也是针对不同的类型进行不同的聚合，所以也相当于采取了不同的模型，而本文并没有将这种GCN-based的方法进行对比。