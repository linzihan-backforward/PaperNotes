# Bipartite Graph Neural Networks for Efficient Node Representation Learning

------

## Motivation

​	利用GNN来进行网络表示学习已经被研究的十分深入，但是大多数的GNN方法都是针对一般的图结构的，而现实中有一类特殊的图结构：二分图，这类图与一般的图有着鲜明的不同点，两类节点之间往往特征分布是不同的，无法直接应用传统的GNN方法，所以在本文中，我们针对二分图提出一种域一致的、无监督的表示学习方法BGNN，其能够适应二分图的特点并高效的完成训练，得到其他无监督模型无法达到的效果。

## Model

​	在一般的GNN模型定义中，输入是节点的初始特征矩阵X、节点之间的邻接矩阵B，输出为每一个节点的表示H，但是在二分图中X和B都可以分成两个，每一类节点独立考虑，因为边只存在于两类节点之间，所以其输入输出可以更细化为如下形式

![image-20200314101747054](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314101747054.png?raw=true)

​	这样的过程对应域间的信息传递过程，那么我们本域内的节点信息怎么利用呢，因为两个域的特征分布是不同的，所以从另外一个域传递的信息应当能够于当前域对齐，那么本域的节点信息就可以作为一个监督项来对齐传递的信息

![image-20200314102101523](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314102101523.png?raw=true)

​	经过这样的两步便能够特例化GNN，得到一个处理二分图的方法，同时域间的信息传递仅仅考虑了一跳的拓扑结果，采用多层串联的方式就可以融合高阶信息，这种串联于传统的多层端到端的方式有所不同，下面会介绍。

### 域间信息传递

​	这一步于基本的GCN过程没有区别太大的区别，对于两个域分别使用对方的特征矩阵和邻接矩阵，聚合一阶邻居的信息，但是没有考虑自环

![image-20200314102633858](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314102633858.png?raw=true)

其中，B同样需要归一化处理

![image-20200314102654121](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314102654121.png?raw=true)

### 域内对齐

​	这一步的作用是训练一个判别器来分辨聚合一阶邻居得来的表示于本域所固有的节点表示，具体地对于上一步得到的Hvu矩阵，我们随机选取一些行，并从原始的矩阵Hu中选择对应行，然后对二者进行0/1分类，这样实际上上一步的信息传递过程充当生成器的角色，两个步骤相互对抗，最终达到均衡，使得结果融合了两个域的信息。那么判别器的Loss就是一个二分类

![image-20200314103722376](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314103722376.png?raw=true)

​	而生成器的Loss就是让判别器预测错误

![image-20200314103805769](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314103803578.png?raw=true)

​	这样的域内对齐当时是一种生成对抗的方式，让二者尽量接近，另外一种更简单直接的方式就是把二者映射到相同维度，然后指定让向量的距离相近

![image-20200314104020880](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314104020880.png?raw=true)

​	这两种都是可行的实现思路，在实验中会对比二者的表现。

### 串联深层结构

​	![image-20200314104946929](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314104946929.png?raw=true)

​	串联结构即将每一层分开训练，每次只训练一层的结构，收敛之后得到的输出作为聚合了一阶信息的表示再重新作为输出进行第二次的训练，这样反复重复完成深层信息的提取，这样的结构训练速度更快同时使用内存更少。在实验中，发现这样的结构于传统的端到端的深层结构相比具有四个优势：内存占用低、不受邻居扩展导致的输入图增大的影响，更快地收敛速度，对于超参数更鲁棒

## Experiments

### Datasets

​	一个全新的腾讯数据集，用户和社区的二分图，三个从引用图中构造的二分图，将节点分为两类，使用不同的特征并移除同类节点的边。

![image-20200314112730117](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314112730117.png?raw=true)

### Baselines

- ​	**Raw features**
- ​    **Node2Vec**
-    **VGAE**
-    **GraphSAGE**
-    **BGNN-MLP** 

### Results

![image-20200314113240854](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Bipartite%20Graph%20Neural%20Networks%20for%20Efficient%20Node%20Representation%20Learning/image-20200314113240854.png?raw=true)

## 个人见解

​	这篇文章整体感觉非常清晰流畅，出发点简洁明了，就是改造GNN来专门处理二分图，思路是将两个域分开对称处理，每一个域聚合对面域的信息，其主要的两点有两个，一个是巧妙的将GAN于GNN结合，将生成式的模型应用于图是非常早就有的思想，聚合邻居节点来生成中心节点也是常见的思路，但是本文的创新是将其于GNN结合从而使得有监督的GNN方法能够无监督训练，同时G和D的两个角色正好与图的两个域对应起来。第二个是分层训练的方法，这个是真的让我学到了，第一次见到这样的训练方式，先不论这样训练能不能得到更好的结果，就单单是这个脑洞我都真的甘拜下风。