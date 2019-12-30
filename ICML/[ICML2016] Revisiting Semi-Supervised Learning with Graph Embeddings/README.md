# Revisiting Semi-Supervised Learning with Graph Embeddings

------

## Motivation

​	网络表示学习（GE）是由NLP中词表示所延申过来的一种技术，其目的是把具有图结构的信息使用低维的实数向量来表示以更方便的进行具体的应用（如节点分类，连接预测等）。已有的GE方法大都是采用的预训练的思路，即使用无标签的数据进行无监督训练，再将结果在下游任务中微调，这种方法适应性广，但是没有充分利用下游任务中的标签信息。所有本工作便提出一种半监督的方法，同时使用有标签和无标签的数据来学习节点表示，将标签信息和网络结构信息一同建模。同时，本方法还可以简单的变形以处理冷启动问题，即得到训练图中未出现的节点的低维表示。

## Model

​	半监督的方法导致数据中存在有标签子集L和无标签子集U，定义两个集合中数据项均包含特征向量x，L集合中另外包含一一对应的y，即节点的标签，目标是为每一个节点学习到其低维表示e。

### Sampling

​	与主流的方法一样，拿到图之后首先要在图上进行采样，得到可以直接输入进神经网络模型的结构化数据，在无监督的方法中采用的是随机游走后的Skip-gram模型进行采样，现在因为要加入标签信息，所以需要对基础的随机游走方法进行拓展。一个采样过程如下图所示。

![image-20191230113908455](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2016%5D%20Revisiting%20Semi-Supervised%20Learning%20with%20Graph%20Embeddings/image-20191230113908455.png?raw=true)

​	整个采样过程被分成了两部分，基于图结构采样和基于标签采样，基于图结构采样即是传统的Deep Walk方法，随机游走之后在窗口内选择中心节点和上下文节点，然后使用中心节点来预测上下文节点，基于的理论假设是：图中位置相近的节点应具有相近的表示。同样，我们可以使用相似的理论假设：标签相同的节点应具有相近的假设，基于这样的假设采取的采样是随机选取一个中心节点和一个上下文节点预测其是否属于同一个类别。两种采样方法由一个阈值r2统一到一起，每次根据随机值采取其中的一种采样策略形成图上的语料库。

​	将Skip-gram的目标函数进行格式化修改得到无监督的Loss![image-20191230115039034](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2016%5D%20Revisiting%20Semi-Supervised%20Learning%20with%20Graph%20Embeddings/image-20191230115039034.png?raw=true)其中的γ表示上下文为正样本还是负样本
​	利用这样的方法便可以得到所有图中节点的embedding，在有监督模型的部分，根据是否能够冷启动，给出了直推式和归纳式两周模型，下面分别说明。

### Transductive

​	直推式模型无法处理冷启动，因为其必须用到无监督部分学习到的节点表示e，无监督方法学到的e被存储于表中供有监督部分查找使用。整个模型架构如下图中的(a)所示。

![image-20191230141138554](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2016%5D%20Revisiting%20Semi-Supervised%20Learning%20with%20Graph%20Embeddings/image-20191230141138554.png?raw=true)

​		两个FC网络分别对特征x和表示e进行映射，至同一空间后再连接起来经过FC得到目标类别输出，有监督部分的Loss通过Softmax来定义<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2016%5D%20Revisiting%20Semi-Supervised%20Learning%20with%20Graph%20Embeddings/image-20191230141441080.png?raw=true" alt="image-20191230141441080" style="zoom:50%;" />

### Inductive

​	归纳式模型可以计算无监督部分未训练过的节点，对于每一个节点其输入都为x，将原本的查表得到的节点表示e更改为一个函数表示f(x)，这样只要知道节点的特征向量x就可以计算出其表示e=f(x)。这种方法本质上是将原本无法进行梯度传播的查表运算改成了可以求提督的DNN网络，也将e从任意随机空间限制在了特征向量x所限制的目标空间，但由于DNN强劲的表示能力，这种空间上的缩小可以忽略。

​	架构如上图中的(b)部分，使用一个额外的FC来充当编码函数f，f(x)作为e的替代来完成无监督和有监督模块中的后续计算，将所有的参数均变为可导参数，整个模型变成端到端的方式。

### Training

​	有监督、无监督两个Loss交替优化。

## Experiments

### Datasets

​	利用得到的embedding，分别选择了文本分类、实体抽取、实体分类这三个下游任务的5个数据集：CITESEER、CORA、PUBMED、DIEL、NELL。

### Results	

​	再不同任务中归纳式和直推式均高于其他的无监督或非图baseline，总体上归纳式效果更好。

![image-20191230143244934](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2016%5D%20Revisiting%20Semi-Supervised%20Learning%20with%20Graph%20Embeddings/image-20191230143244934.png?raw=true)

## 个人见解

​		这篇文章年份偏早应该是Deep Walk之后第一批在此基础上拓展的模型，整体的创新点是利用有标签数据来微调无监督模型以及将embedding于feature相联系来解决冷启动，在了解了最近一些相似的模型之后不免觉得这样的网络结构和两段式的结合方式过于简单，但其在近5年前也是半监督图嵌入的典型方工作了，如果站在现在的视角看，模型设计上和实验上还有很多不足的地方，如节点输入特征x如何得到，在特征不明显的网络上归纳与直推相比怎样，为什么采用交替训练而不是一同优化的方式等。