# Conditional Network Embeddings

------

## Motivation

​	在已有的网络嵌入方法中，整体的模型可以分为三步，首先定义在图上的相似度，然后定义在嵌入向量空间的相似度，最后定义损失函数让两个相似度想接近。这种方法假设在欧氏空间中能够表达图上的结构，但是某些网络特征，如连接的多分性，节点度的分布等复杂的信息往往不能表达在这些方法中，所以本文提出一种条件网络嵌入，最大限度地增加结构的信息（如节点度，块密度）并使用简单的贝叶斯方法来实现。

## Model

​	让节点的嵌入矩阵包含更多的原始图的信息可以写作一个极大似然估计的形式：

![image-20200402145739668](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402145739668.png?raw=true)

​	上面的式子是不好直接进行求解的，我们可以使用贝叶斯法则来引入一些先验知识P（G）![image-20200402150025605](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402150025605.png?raw=true)

有了这个转化，还需要分别定义网络的先验分布和数据的条件分布两方面，首先是P（G）。我们考虑三类图中的信息网络的整体密度、单个节点的度、指定子集内或自己之间的边缘密度信息。这三类信息都可以表示为一种边的集合，而假设每一条之间互相独立服从伯努利分布，那么对于一个边的集合，其对应的图先验知识可以表达如下：

![image-20200402151412788](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402151412788.png?raw=true)

​	对于数据的条件分布，因为我们仅仅关心节点之间的距离，所以生成X的概率应当仅仅与网络G中的节点距离有关，我们让存在连接的节点之间的距离服从一个方差更小的半正太分布，而不存在连接的服从方差更大的正太分布，这样就能够限制其邻居之间的距离的期望更近

![image-20200402152431504](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402152431504.png?raw=true)

最后，后验分布可以使用上面定义的计算得到

![image-20200402152617864](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402152617864.png?raw=true)

最后，将其全部带入贝叶斯公式中，使用最大似然估计的方法就能够得到满足最大化上面条件概率的一个X

## Experiments

### Datasets

![image-20200402153507578](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402153507578.png?raw=true)

### Baselines

### Results

![image-20200402153636090](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20Conditional%20Network%20Embeddings/image-20200402153636090.png?raw=true)

## 个人见解

​	本文是希望将一些先验的知识启发式的加入到图嵌入的过程中，通过贝叶斯的方式定义图的先验知识为边的伯努利分布，然后再为嵌入过程增加限制。其思路是让邻居和非邻居节点服从到两个不同的正态分布上面。从实验结果来看其将其他方法甩的很远，但是其又加入了额外的启发式知识，也就是说模型并没有完全非监督的得到节点的表示，而是委婉的加入了一些辅助监督信息，再加上其理论部分占比过多导致实际实现时很多还有很多问题并没有说清，而且其缺乏和深层模型的对比，所以这种贝叶斯的思路我觉得并不是一个值得发展的思路，因为其有伪无监督的嫌疑。