# Semi-supervised User Profiling with Heterogeneous Graph Attention Networks

------

## Motivation

​	用户画像是各类在线购物和社交平台的一项核心技术，其希望利用用户的属性和行为为其建立一个独一无二的表示，从而应用在下游任务中。已有的很多方法将其建模为一个分类任务，即通过输入的特征得到用户的分类，这样实际上割裂了用户之间的关系。我们希望将用户之间的连接关系一并建模到用户的画像中，利用用户、商品、属性，三层的异质图来表征全部的连接关系，并使用GAT来得到节点（用户）的表示。

## Model

​	用户画像指的是对用户的特征进行预测，其中一个代表便是对用户性别和年龄预测，这里的半监督指的是我们仅仅用户一小部分用户的年龄标签，同时用用一个异质的用户图，希望利用图上的连接关系作为若监督信息来加强模型。首先，需要定义我们的异质图，图中包含三类节点，即用户、商品、属性，三类边，用户与用户的边、用户和商品之间的边、商品与属性的边。其含义在具体地应用场景中都可以轻易找到。

![image-20200303151923694](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Semi-supervised%20User%20Profiling%20with%20Heterogeneous%20Graph%20Attention%20Networks/image-20200303151923694.png?raw=true)

​	模型的第一步少不了进行Embedding，在这里我们的图分为三层，上层是基于下层的，所以我们仅仅需要对底层的属性节点进行embedding，上层的embedding由模型学到，这里假设我们的属性节点为词节点，所以我们定义每一个属性节点的初始embedding为使用Fast-Text在所有语料上学习到的词的表示。

​	在有了节点的初始嵌入之后，我们考虑使用图中的邻居节点来更新节点的嵌入，这里就有了三种可选的更新方式，传统的注意力机制，GCN，GAT，其中传统注意力与GAT的区别在于其只使用邻居节点的嵌入来计算权值。这三种方法都是比较常见的方法了，其详细的计算公式这里就不再介绍。

​	有了节点的更新方法之后，还有一个主要的问题没有解决，那就是如何在不同类型的节点之间传递信息，这也是我们的方法与前面提到的那些直接在同质图上更新的方法的主要区别。

​	我们的边共有三种类型，那么针对每一种不同的边，我们采取不同的传播方法，又由于其层次性，所以我们的传播过程也可以分成三种进行，首先从属性节点到商品节点，然后从商品节点到用户节点，最后用户节点之间。这三种我们采用不同的聚合方式，整体如下图所示。

![image-20200303154722667](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Semi-supervised%20User%20Profiling%20with%20Heterogeneous%20Graph%20Attention%20Networks/image-20200303154722667.png?raw=true)

​	在得到了用户的表示之后，我们可以使用用户向量再进行softmax之后进行多分类任务，最终的目标使用cross entropy函数

![image-20200303155325184](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Semi-supervised%20User%20Profiling%20with%20Heterogeneous%20Graph%20Attention%20Networks/image-20200303155325184.png?raw=true)

## Experiments

### Datasets

​	使用的数据集来自JD，一个全新的数据集，包括用户、商品、商品的标题，将标题中的词作为底层的属性构图，预测的目标是用户的年龄和性别。

![image-20200303181047477](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Semi-supervised%20User%20Profiling%20with%20Heterogeneous%20Graph%20Attention%20Networks/image-20200303181047477.png?raw=true)

### Baselines

- ​	**LR**

-  **SVM**

- **GCN**

- **GAT**

  

### Results

![image-20200303181219435](https://github.com/linzihan-backforward/PaperNotes/blob/master/IJCAI/%5BIJCAI2019%5D%20Semi-supervised%20User%20Profiling%20with%20Heterogeneous%20Graph%20Attention%20Networks/image-20200303181219435.png?raw=true)

## 个人见解

​	本文最大的创新点在于对异质图中三种类型的节点按照层次的顺序分层更新，不同层之间采取不同的更新策略。在任务和实现细节方面使用的是一些成熟的方法。这种用户画像的任务相比较于推荐将关注点进一步缩小，从用户-商品二元关系转变为了仅仅关注用户，这样的视角的缩小有利于突出模型的重点，同时也使得方法上work的可能性更大，针对目前所做的工作，这样的视角缩小是一个值得探讨的点。本文使用的是一个JD的全新数据集，这个数据集或者代码如果能够放出的话会是一个新的baseline