# End-to-End Structure-Aware Convolutional Networks for Knowledge Base Completion

------

## Motivation

​	知识图谱补全是知识图谱中一个重要的研究点，其中一个关键的计数便是知识图谱的嵌入，从最初的TransE，TransH到最新的ConvE，各种嵌入计数通过不同的方法或者不同的嵌入空间来更新方法，但是这些已有的方法都一个共同的局限性，即没有考虑结构信息，实体周围连接的结构信息同样是一个很重要的特征，但是之前的方法将各个关系分开考虑，无法对整体的结构进行把握，而GCN作为最新的处理图结构的利器便可以用来克服这个局限，所以本文通过将GCN和ConvE两种方法的优势互补，来提出一种新的考虑结构信息的知识图谱嵌入方法。

## Model

​	我们的模型分为两个部分，第一部分学习实体（节点）的嵌入，第二部分学习关系的嵌入，两部分连接起来可以进行端到端的训练。

### Weighted Graph Convolutional Layer

​	WGCN可以看作是传统的GCN的一个变型，它将原本知识图谱按照存在的关系数目分为若干个子图，在每一个子图中，使用GCN的方法来聚合周围的邻居，然后对于所有的子图再通过一个权值α聚合起来，对于图中的节点i其一层WGCN的计算如下

![image-20200222090530505](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222090530505.png?raw=true)

其中α为连接t所对应的节点所贡献的一个权重。尽管对于函数g有很多种实现实现方式，我们这是选择的是最简单的与节点i无关的实现方式，仅仅使用一个线性映射将输入的向量映射到输出向量。

![image-20200222090938001](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222090938001.png?raw=true)

​	代入原始的表达式中可以发现每一层是对邻居的线性求和，为了保持结果的稳定，我们加入一个自环，并将其权重固定为1

![image-20200222091208428](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222091208428.png?raw=true)

将图的邻接矩阵按照连接类别乘以权重之后，上面的计算方式便能够与GCN相统一

![image-20200222091445008](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222091445008.png?raw=true)![image-20200222091504689](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222091504689.png?raw=true)

### Conv-TransE

​	在通过GCN的方式得到了节点的嵌入之后，第二步的目的便是利用节点的嵌入得到关系的嵌入，并且保持实体和关系之间的可加性。受到ConvE方法的启发，我们同样采用卷积的方法，但区别是不进行整形操作。

​	将节点的嵌入和关系嵌入拼接之后，得到2×L的一个矩阵，之后使用多个2×K的卷积核对其进行一维的卷积，输出为1×L的feature map，因为使用的是一维卷积，所以在两行之间使用的是相加关系，能够保持其线性可加性，将所有C个卷积核得到的计算结果凭借起来，得到1×CL的一个大向量，再线性映射到1×L即完成了这一部分的计算

![image-20200222094607213](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222094607213.png?raw=true)

结果直接与目标实体eo进行点积以得到最终的分数。

模型的整体示意图如下

![image-20200222094801120](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222094801120.png?raw=true)

## Experiments

代码开源：https://github.com/JD-AI-Research-Silicon-Valley/SACN

### Datasets

​	FB15k-237

​	WN18RR

### Baselines

​	DistMult

​	ComplEx

​	R-GCN

​	ConvE

​	Conv-TransE

### Results

​	![image-20200222095415438](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2019%5D%20End-to-End%20Structure-Aware%20Convolutional%20Network%20for%20Knowledge%20Base%20Completion/image-20200222095415438.png?raw=true)

## 个人见解

​	本文的目标是知识图谱补全，具体的实际上模型是一个知识图谱嵌入的方法，将原本一同学习实体和嵌入的方法分割成了先实体后方法两个部分，其中第一部分将热门的GCN方法应用在了这里，先分类，再根据不同权重进行线性聚合的方法在多个工作中都被应用，通过将各类GCN、GAT方法再套上这么一层权重，又诞生了一系列的处理异质节点的方法，不能说方法不好，但是总感觉有一点灌水的意思，思想上大同小异，拼的就是结果和时间了。本模型的第二部分同样使用的是一个之前工作中的方法，两部分虽然说是各自分工但是感觉彼此割裂，找不到道理上相连的解释，只能说连起来比一个的效果好，启发性不够。