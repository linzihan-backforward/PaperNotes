# FastGCN_Fast_Learning_with_Graph_Convolutional_Net

------

## Motivation

​	在最基本的GCN中，因为是在谱空间的运算，所以就必须对整个图进行整体的卷积，但是这样就限制了其在大图上的应用，为了能够将其拓展为批训练的方式以及使其能够处理未见到的数据，我们将卷积理解为一种积分变换并使用蒙特卡洛方法来近似以达到加快训练的作用，正如题目中的名称一样，这就是一个更加Fast版本的GCN

## Model

​	在GCN中一个显著的性质不再满足：数据之间不相关。在其他场景中，由于这个性质使得我们可以进行批训练，而图中由于每一个节点周围的邻居被牵扯进其目标函数中所以就不再满足，传统的GCN的公式可以表示如下

![image-20200401103418760](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20FastGCN_Fast_Learning_with_Graph_Convolutional_Net/image-20200401103418760.png?raw=true)

​	我们从积分的角度理解这个公式，即每一个节点满足一个概率分布，聚合邻居节点的过程看作是一个求积分：

![image-20200401103716393](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20FastGCN_Fast_Learning_with_Graph_Convolutional_Net/image-20200401103716393.png?raw=true)

​	这样的话，可以使用蒙特卡洛方法从所有节点中采样来近似这个积分。

![image-20200401104017882](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20FastGCN_Fast_Learning_with_Graph_Convolutional_Net/image-20200401104017882.png?raw=true)

​	这样的话每一层只需要采样除tl个节点进行卷积运算，极大的减少了计算量同时可以进行批训练。这样的方法其最终的Loss可以由大数定理逼近最终真实的Loss，而每一层产生的偏差都是在一个固定范围，可以在原文中找到相关证明。

## Experiments

### Datasets

两个学术引用网络数据集与一个社交网络数据集

![image-20200401143028042](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20FastGCN_Fast_Learning_with_Graph_Convolutional_Net/image-20200401143028042.png?raw=true)

### Baselines

- **GCN**
- **GraphSAGE**

### Results

![image-20200401143748492](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20FastGCN_Fast_Learning_with_Graph_Convolutional_Net/image-20200401143748492.png?raw=true)

## 个人见解

​	本文应该是和GraphSAGE基本同时的文章，从文中可以看到作者也是在基本完成之后才发现又出现了一个非常类似的GraphSAGE，然后加入对比的，二者效果上面互有高低，基本类似，就不在过多评价。其思想上面与GraphSAGE完全不同的角度，其还是专注于谱域的卷积，将卷积看作一种特殊的积分运算，然后使用积分上的近似来解决采样问题，如果说GraphSAGE更加简单直接的话，那FasGCN就延续了GCN优秀的理论基础，二者各有优势，只不过由于实现上和理解上略微复杂，其在后续的工作中出现的并没有GraphSAGE那样多。