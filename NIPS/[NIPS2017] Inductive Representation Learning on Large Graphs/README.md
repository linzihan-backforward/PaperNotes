# Inductive Representation Learning on Large Graphs

------

## Motivation

​	已有的网络表示学习方法大多数都是直推式的，即只能得到训练集中出现过的节点的表示，而无法处理完全全新的节点，更抽象的说法便是无法处理冷启动问题，仅有的对这些方法的改造也需要大量的计算（梯度下降）来处理新加入的节点，针对这样的问题，我们便提出了一种使用节点特征的方法，其能够自动学习邻居节点信息的聚合方法，并同时刻画邻居的拓扑结构和特征数据。这种方法也可以看作GCN的一种拓展，因为其聚合方式不仅仅局限于卷积的方式。

## Model

​	模型的主要思想是学习如何聚合一个节点周围邻居节点的特征，这是使用的是具有特征信息的图，即每个节点有一个额外的特征向量，但对于没有特征的图，依然可以使用（如将节点的度作为特征）。

### 表示生成

​	首先，我们假设已经得到了k个聚合函数，其可以将输入的节点集合进行聚合，得到一个单一的输出，具体聚合函数的实现放在后面小节讨论，则我们的整个表示生成过程如下：使用每个节点的特征向量作为初始的表示，然后进行k次表示聚合即循环k次，循环中对于每一个节点使用公式得到其本层的表示，将第k层的表示，也就是循环结束后得到的表示作为模型的输出。聚合的公式在下图算法流程中。

![image-20191231120341943](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Inductive%20Representation%20Learning%20on%20Large%20Graphs/image-20191231120341943.png?raw=true)

​	其中AGGREGATEk为第k次循环采用的聚合函数，其输入为一个向量集合，输出为单独一个向量，Wk为第k层的参数。这种循环聚合的过程与图的同构检测算法非常相似，只不过将其中不可导的哈希函数变成了神经网络聚合器。

### 损失定义

​	前向传播得到z之后，为了训练还需要定义一个损失函数，与Deep Walk 一样，使用正样本和采样得到的负样本进行分类损失，即Skip-gram方法的损失。区别是我们的z为上述前向传播过程计算得到，而非传统的查表得到。

![image-20191231142305596](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Inductive%20Representation%20Learning%20on%20Large%20Graphs/image-20191231142305596.png?raw=true)

### 聚合函数

​	这里的聚合函数目标是将一个节点的邻居信息聚合成一个向量，所以其应当是顺序不敏感的，输入顺序不会导致输出发生变化，所以这里分别使用了均值、LSTM、最大值池化三个聚合函数。

​	均值即对输入的所有向量求平均，LSTM严格意义上是顺序敏感的，但因为其丰富的表达性，这里依旧使用，最大值池化即对于每一个输入的参数让其通过一个单层的FC，然后将所有的输出再进行Max Pool。![image-20191231143247365](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Inductive%20Representation%20Learning%20on%20Large%20Graphs/image-20191231143247365.png?raw=true) 



## Experiments

​	作者开源了其代码实现，工程化做的比较好：https://github.com/williamleif/GraphSAGE

### Datasets

​	作者使用了三个不同领域的数据集来进行实验：论文引用数据集来预测论文分类，Reddit帖子数据集来进行帖子主题分类，蛋白质关系数据集来预测蛋白质功能。三个数据集均使用新节点信息作为测试集。

### Results

![image-20191231145955039](https://github.com/linzihan-backforward/PaperNotes/blob/master/NIPS/%5BNIPS2017%5D%20Inductive%20Representation%20Learning%20on%20Large%20Graphs/image-20191231145955039.png?raw=true)

## 个人见解

​	本文的主要贡献应该就是将GCN方法进行了一般化并且与Deep Walk等非监督的图嵌入方法进行了统一，同时其宣称能够缓解冷启动问题，但是从整体的模型来看，其只是实现了“伪”缓解，对于未出现过的数据还是需要训练集中的数据来进行协助，与其他的能够处理冷启动的方法相比可能就是计算量上的差别，并没有本质上的进步，真正的缓解冷启动应当能够抽象为一个端到端的函数f（x）其只与输入的测试数据有关而不应当包含其他的输入，并且参数再充分训练后应该能够固定。同时这篇工作作为17年的模型其实验对比项选取的也有些问题，缺乏类似模型的对比而只有最基础的模型，说服力上比较欠缺，但是其作为GCN的泛化版本，在后续此类方法的变形改进上还是有一定的启发作用。