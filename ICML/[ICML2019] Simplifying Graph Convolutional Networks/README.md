# Simplifying Graph Convolutional Networks

------

## Motivation	

​	在最近几年深度学习大火之前，大部分的模型都是简单的线性模型，它们在一些问题上也能提供一定的作用，但是随着任务的复杂性越来越高，对性能的要求越来越高，大量非线性的深度神经网络模型走上了前沿的舞台，它们在提升了性能的同时也一并带来的大量的计算和不可解释性，配套的理论到目前依旧难有突破。反观在图领域，GCN是随着深度学习的发展而被提出的，其在最初就被设计为非线性的，这与其他的循序渐进的方法有所不同，所以本文希望能够回到GCN之前，找到一个更简化版本的图卷积操作，其应当是纯线性的，并检测这样的线性模型与其他模型的区别和优势。

## Model

​	在介绍线性的图卷积模型之前，让我们先回顾一下传统的GCN的流程。对于每一个图中的节点，初始有一个对应的特征向量xi，将图中所有的特征向量堆叠形成特征矩阵后X作为模型的输入，在经过k层结构相同的卷积层之后得到的表示矩阵H作为每个节点新特征向量来喂给下游的分类任务。

​	在每一层的卷积中，包含三个部分：特征传播，线性转换，非线性激活。特征传播即对于每一个节点将其直接相连的邻居的特征聚合起来与本节点上一层的特征相连或相加，线性转换通常是使用一个全连接的神经网络来对聚合得到的特征进行映射，非线性激活则对应全连接网络最后的非线性激活函数（ReLU），整个的GCN过程如下图所示。

![image-20200106101809943](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Simplifying%20Graph%20Convolutional%20Networks/image-20200106101809943.png?raw=true)

​	其中每一层的运算都使用了整个图的矩阵来直接进行矩阵运算，特征聚合的运算矩阵S可以由图的邻接矩阵和度对角矩阵得到。![image-20200106101931385](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Simplifying%20Graph%20Convolutional%20Networks/image-20200106101931385.png?raw=true)

### Simple Graph Convolution

​	模型基于一个很简单的假设，即非线性激活运算在模型中的作用不大，主要起作用的操作是特征传播运算，所以我们直接在每一层的卷积中去掉最后的ReLU运算，则我们的整个模型完全变成了一个线性的模型。

![image-20200106102900241](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Simplifying%20Graph%20Convolutional%20Networks/image-20200106102900241.png?raw=true)

​	显然，其中的连续的矩阵相乘项可以表示为一个矩阵。

![image-20200106102955060](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Simplifying%20Graph%20Convolutional%20Networks/image-20200106102955060.png?raw=true)

​	这样我们的模型由k层压缩为了一层，S矩阵非模型参数，所以相当于使用图的邻接矩阵进行特征预处理之后再进行一个线性映射便得到了模型输出，这样的话整个模型的速度便得到了极大的提高，在下面的分析和实验中，将展示这种方法与非线性的方法相比的效果。

​	在理论分析部分，本文从图的傅里叶变换角度证明了特征传播中的自环连接部分在线性模型中充当了一个地通滤波器的作用，碍于公式的繁多和本人水平有限，对此部分由需求的话可以查阅原论文中详细的描述。

## Experiments

​	本实验的代码开源：https://github.com/Tiiiger/SGC

### Datasets

​	本文使用了两类数据集，学术引用网络和社交网络。其中学术引用网络包含：Cora、Citeseer、Pubmed，社交网络为Reddit。在这些数据集上进行的任务都是半监督的节点分类任务。

### Baselines

​	这里使用的baseline都是GCN及其变形，不再详细介绍，包括：GCN、GAT、FastGCN、GIN、LNet、AdaLNet、DGI。

### Results

![image-20200106121453997](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICML/%5BICML2019%5D%20Simplifying%20Graph%20Convolutional%20Networks/image-20200106121453997.png?raw=true)

​	本文的实验部分非常充分，除了之间的节点分类实验之外，还测试了5个不同具体下游任务，其结果可以去原文中查看。

## 个人见解

​	这篇文章的角度和思路上还是非常新奇的，有点不按常理出牌的意思，在大家模型都朝着越来越复杂走的时候把一个如此简单甚至不能称之为模型的方法拿出来，同时还有做实验来验证的勇气，真的是让我吃了一惊吧，构造模型的角度应该算图嵌入方向比较新奇的几篇之一了。话说回来，如果确实仅仅使用一个线性的特征变换和线性的参数就可以达到近似的效果的话，是不是意味着近几年关于GCN的各种研究都走进了一个误区：GCN的核心作用点不在于个性化的聚合图中的节点，而只是利用到了更多的有关系的数据所带来的泛化能力。如果确实是这样的话，那么还有很多的证明工作要做，整个GCN的研究方向也要收到影响，传统的同样结构堆积多层的方法可能不再好做了。个人觉得这篇文章在GCN方向还是有一定的影响作用的，也启发了我们不能墨守成规，多放开思路，不要被大方向裹挟，一些大胆的想法有可能也能work。