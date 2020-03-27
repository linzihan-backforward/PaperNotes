# Semi-supervised Classification with Graph Convolutional Networks

------

## Motivation

​	在各类数据中图结构的数据占比非常多，文本提出了一种半监督的方法，将变体的卷积神经网络作用于图数据来学习到节点的嵌入，此嵌入既包含节点的特征信息又包含局部的结构信息并且此方法于图的边数为线性的关系。

## Model

​	之前的模型中往往都在目前中加入了一个图拉普拉斯正则项，其假设图中有边连接的节点可能会具有相近的标签，所以将两点的二阶距离作为目标，但是这样的假设可能会限制模型的能力，因为边能够提供的不止是这个信息。所以我们提出一种多层的GCN每一层使用如下的传播规则：
$$
H^{(l+1)} = \sigma (\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}) 
$$
其中A为邻接矩阵加入自环之后的矩阵 D为对角矩阵，每个值为节点的度，H为当前层的表示，初始化为特征矩阵X。

这个传播规则可以由图上一阶局部近似以及切比雪夫滤波器简化得来，具体的推导这里省略。

在一个多层的GCN中，H之前的部分充当一个选择器的作用，其可以通过邻接矩阵预处理出来，而后面的W为每一层的模型参数，使用两层的GCN来进行节点的预测时，只需要W0和W1两个参数矩阵即可

![image-20200326110900476](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2017%5D%20Semi-supervised%20Classification%20with%20Graph%20Convolutional%20Networks/image-20200326110900476.png?raw=true)

得到的Z便是每一个节点的嵌入表示，分类任务中需要在那些有标签的节点使用Cross-entropy的Loss

![image-20200326111015117](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2017%5D%20Semi-supervised%20Classification%20with%20Graph%20Convolutional%20Networks/image-20200326111015117.png?raw=true)

注意：这里需要对整个图进行预处理，所以必须知道全部的图信息，每一次也必须使用全部图的特征矩阵X进行训练。

## Experiments

### Datasets	

使用了三个引用网络和一个知识图谱网络。

![image-20200326115009209](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2017%5D%20Semi-supervised%20Classification%20with%20Graph%20Convolutional%20Networks/image-20200326115009209.png?raw=true)

### Baselines

- **LP** label propagation
- **SemiEmb** 
- **manifold**
- **DeepWalk**
- **Planetoid**

### Results

![image-20200326145540692](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2017%5D%20Semi-supervised%20Classification%20with%20Graph%20Convolutional%20Networks/image-20200326145540692.png?raw=true)

## 个人见解

​	本文是GCN的开篇之作，也是质量非常高的一篇，从基本的一阶滤波器入手得到一个加入自环并用度来正则化的过滤项，关键是这一项还可以提前预处理出来，使得整个训练过程就需要简单的一个参数矩阵和矩阵乘法就可以达到，真是结构和结果同样优美，尽管本文并不是将所有问题都解决，但是给出了一个框架，指向性非常清晰，也不枉其带火了一个新的方向。