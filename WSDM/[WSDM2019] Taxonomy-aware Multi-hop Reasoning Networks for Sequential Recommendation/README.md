# Taxonomy-aware Multi-hop Reasoning Networks for Sequential Recommendation

------

## Motivation

​	序列推荐是推荐系统的一个最典型的形式，给定用户历史的交互记录，目标是预测下一个出现的商品，随着相关技术的发展，有很多工作使用额外的上下文信息来建模这个序列，尽管有一些效果，但是仍有一些局限性待解决，如得到的序列特征单一，无法包含细粒度信息，上下文数据利用不明确等。针对这些问题，我们希望利用一个定义好的商品分类学特征来辅助序列建模，在可解释的前提下捕捉细粒度的用户偏好。

## Model

​	在电商平台中，一个商品具有一个从宏观到具体地分类信息，如MacBook的分类为电脑办公 ![[公式]](https://www.zhihu.com/equation?tex=%5Crightarrow) 轻薄本 ![[公式]](https://www.zhihu.com/equation?tex=%5Crightarrow) MacBook ，这样所有的类别标签能够形成一棵树，树的叶子节点代表商品，非叶子节点代表分类标签，在得到整个树之后，我们希望将用户的偏好学习与树中的每一层相对应，即用户从根节点开始，每次选择一个孩子作为当前的偏好，直到走到叶子节点完成购买，这样就能够将抽象的用户的偏好具体化，整个学习和推理过程变得具有可解释性。一个用户分层的商品选择过程如图

![image-20200219094541093](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219094541093.png?raw=true)

​	本模型分为两个部分，一个GRU来建模用户的序列行为，一个记忆网络来进行多跳的推理，两个部分的输出共同作为用户当前的偏好。

​	GRU的部分与其他方法中用到的相同，将序列的商品嵌入依次通过GRU模型，最终的隐层向量作为用户的偏好表示。

​	记忆网络可以使用读取操作对存储的记忆进行查询和修改，假设这个分类树存在K个分类层次，则我们使用K个独立的记忆单元（矩阵）来分别保存这K次选择的偏好，注意，每一个用户是独立的，对于所有的用户|U|，需要|U|×K个矩阵来作存储。

​	细粒度的偏好应当由粗粒度的偏好中得到，所以用户u在第k跳时的偏好可以通过第k-1跳的偏好和当前跳记忆网络的输出决定。

![image-20200219100437341](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219100437341.png?raw=true)

其中v表示用户的偏好表示，o表示一个记忆单元的输出值，同时这个输出值应当使用上一层的偏好表示从当前记忆矩阵M中读取出来。

![image-20200219100623853](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219100623853.png?raw=true)

那么我们的记忆矩阵M时如何定义的呢，我们将其定义为A个向量的集合，其中每一个向量由一个商品的嵌入和此商品对应的当前层次的分类标签的嵌入组成，这个标签的嵌入使用LINE模型预训练。

![image-20200219101238373](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219101238373.png?raw=true)

​	通过这样的定义，我们可以看到每一个矩阵能够存储A个商品的信息，也就是对于一个用户只存储其最近的A次交互的信息，当有一个新的交互产生时，需要用新的商品表示来替换矩阵中最老的那一个向量，这样的操作可以使用队列进行实现。

![image-20200219101431681](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219101431681.png?raw=true)

那么读操作又是怎么具体实现的呢？其实更加简单，通过查询和矩阵中的每一个向量之间的注意力机制便能够实现。

![image-20200219101757508](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2019%5D%20Taxonomy-aware%20Multi-hop%20Reasoning%20Networks%20for%20Sequential%20Recommendation/image-20200219101757508.png?raw=true)

​	将用户在每一个选择上的偏好聚集起来和传统的GRU生成的偏好一起便可以作为用户整体偏好，即包含粗粒度又包含细粒度。

## Experiments	

​	https://github.com/RUCDM/TMRN

### Datasets

​	使用的三个开源的数据集，方便进行结果对比：Amazon Music、JD、LAST.FM



## 个人见解

​	本文的主要思想是将用户的偏好建模在分类标签上，具体地实现方法使用到了记忆网络来存储用户在每一层的标签上的选择信息，然后利用查询操作来得到当前层上的具体偏好。这种将复杂的决策行为进行模式解析，进而得到子模式上的细粒度偏好的思想可以是一类推荐系统的主要思想，包括使用kg的，本质上都是分解这个偏好来加强效果同时得到解释。本文另外一个创新点便是使用了记忆网络，这种结构被使用的并不多，利用队列先进先出的思想同样非常有启发性，利用额外知识进行推荐的方法其实都可以考虑是不是能够使用记忆网络。