# Neural Graph Collaborative Filtering

------

## Motivation

​	在基于协同过滤的方法中，一个关键的点就是如何获得用户和商品的向量表示（嵌入），不论是早期的矩阵分解的方法还是最近的使用深度学习的方法，得到的用户嵌入都仅仅与用户和商品的原始ID和属性有关，而一个最关键的信号——协同属性并没有被直接使用到嵌入函数中，而仅仅作为使用嵌入建模交互过程中的一个间接的目标函数来优化，这种方式并不能在生成阶段得到高质量的嵌入。针对这一点，我们通过直接将用户-商品二部图中的高阶连接关系引入表示生成阶段，让用户的表示在图中进行传播来直接将协同属性纳入到嵌入中。

## Model

​	![image-20200302154023811](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302154023811.png?raw=true)

​	整体的模型如上图所示，可以看到分为三个模块，其中中间的嵌入传播模块是主要的功能模块，它的作用是将随机初始化的嵌入在图上进行多层的传播，使得其带有图中高阶连接关系的信息，下面着重介绍这个部分。

​	首先，少不了的一步就是为所有的用户和商品建立对应的嵌入表，通过随机初始化的方法将一个用户对应到一个向量。将这个随机得到的表示作为图中节点的初始向量，开始GNN的信息传播过程。

​	对于连接用户u和商品i的一条表（u，i）我们定义从i传递向u的信息如下：

![image-20200302154923942](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302154923942.png?raw=true)

​	这样的传播定义与GCN非常相似，唯一的区别在于将目标节点的信息eu也使用了进来。

​	之后与GCN一样，我们需要定义所有传递信息的聚合方式：

![image-20200302155327483](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302155327483.png?raw=true)

​	这样我们通过对所有的用户和商品节点进行一次上述的信息传播操作，我们得到的输出变带有了一阶的连接关系。将同样的结构输入输出相连，即可得到循环式的多层信息传播，其中每一层共享同样的W1、W2，而不同层之间则使用不同的参数。

​	最终的用户和商品的嵌入使用每一层的结果相连接

![image-20200302160531981](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302160531981.png?raw=true)

​	预测同样使用非常简单的点乘方式

![image-20200302160559622](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302160559622.png?raw=true)

​	最终的目标函数为BPR Loss

![image-20200302160804421](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302160804421.png?raw=true)

## Experiments

### Datasets

![image-20200302190645692](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302190645692.png?raw=true)

### Baselines

- **MF**
- **NMF**
- **CMN** 使用基于记忆的方法合并用户的一阶邻居
- **HOP-Rec** 使用图上的random walk来丰富用户和商品的交互记录
- **PinSage** 在图上使用GraphSAGE的方法
- **GC-MC**   仅使用一层GCN得到图上的用户和商品节点的表示

### Results

![image-20200302191339130](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2019%5D%20Neural%20Graph%20Collaborative%20Filtering/image-20200302191339130.png?raw=true)

## 个人见解

​	这篇文章应该是应用GCN方法在推荐上模型最直接最简单的了，直接在二部图上面应用GCN，虽然问题上和模型上没有什么大的问题，但是总感觉和其他的图上推荐的方法找不到本质上的创新，问题还是老问题，需求还是老需求，只能是强行解释了一下GCN可以刻画高阶的连接，但是其他基于path的方法同样可以有高阶连接，又劣势在哪呢。最近读的几篇何向南老师组的图推荐方向的文章，都给人一种相似的感觉，就是模型方法上不复杂，甚至有些相似，而又有细微的差别。他们的文章可以给人提供不同的baseline，但motivation上的启发又非常有限，见仁见智吧。