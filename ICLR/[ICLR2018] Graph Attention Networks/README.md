# Graph Attention Networks

------

## Motivation

​	将卷积操作应用于图结构数据已经获得了很好的效果，相关的变形模型也多种多样，其中的关键一步便是对数量不定的邻居集合进行聚合的操作，收到Attention思想的影响，我们希望利用其天然的对这种长度不定的输入的适应能力来构建一种区别于卷积操作的图神经网络结构，保留GCN的优势并使用self-Attention作为信息传递的方式，这种新的GAT能够并行化计算并得到SOTA的效果。

## Model

​	我们的问题定义与最基本的GCN相同，将每个结点的特征向量h作为输入，一层GAT的输出应该同样为每一个节点的特征h'，与之前的方法中使用平均值池化或者LSTM来处理邻居集合的特征不同的是，我们希望每一个邻居能够学习到一个权重，然后再使用此权重来作加权平均，所以整个模型的关键便是得到这个权重上，类比于序列上使用的self-attention方法，我们将当前节点的特征hi作为query，将所有的邻居节点作为key，使用softmax的方法来计算集合中的所有节点的value值。

![image-20200118155825647](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20Graph%20Attention%20Networks/image-20200118155825647.png?raw=true)

​	其中W为本层的线性转换矩阵，a代表一个FC网络，用来求输入的分值，再对于所有的邻居集合Ni使用softmax进行分数归一化，这里的W和a均在所有节点之间共享，极大的节省了模型的参数，再计算出权重之后，聚合操作变成了加权求和。

![image-20200118160752198](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20Graph%20Attention%20Networks/image-20200118160752198.png?raw=true)

​	与序列上的attention一样，这里使用多头的注意力也会增加模型的表现能力，将上面的所有参数平行复制K份，在层的最后把结果进行拼接就可以得到多头的结果。

![image-20200118160946205](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20Graph%20Attention%20Networks/image-20200118160946205.png?raw=true)

## Experiments

### Datasets

​	三个学术网络数据集作为直推式任务——Cora，Citeseer，Pubmed

​	一个蛋白质关系网络数据集作为归纳式任务——PPI

### Baselines

​	直推式任务：

- ​	LP（label propagation）
- ​    SemiEmb（semi-supervised embedding）
- ​    ManiReg（manifold regularization）
- ​    DeepWalk
- ​    ICA and Planetoid
- ​    GCN
- ​    MoNet

​	归纳式任务：

- ​	GraphSAGE-GCN
- ​    GraphSAGE-mean
- ​    GraphSAGE-LSTM
- ​    GraphSAGE-pool

### Results

![image-20200119112730505](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20Graph%20Attention%20Networks/image-20200119112730505.png?raw=true)

![image-20200119112750332](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2018%5D%20Graph%20Attention%20Networks/image-20200119112750332.png?raw=true)

## 个人见解

​	读了那么多引用GAT的文章，这总算是把这个GCN之后又一个高质量的文章坑给添上了吧，图灵奖得主挂名的文章质量上肯定过得去，读完发现如果如此，模型上虽然没有太多的太复杂的创新点，但是就是给人一种简单直接有效的感觉，可能是怕模型部分太过少又强行对比了一下GAT和前面模型的异同点，实验部分应该能算上最充分的文章中的之一了，直推式和归纳式两种任务找到接近20种对比项来验证GAT的SOTA效果，足以见到作者在图方面的深厚功底，文章的行文上还是跟华人作者的文章有很大的区别，表达上感觉更加直接又不突兀，值得多读的基础好文。