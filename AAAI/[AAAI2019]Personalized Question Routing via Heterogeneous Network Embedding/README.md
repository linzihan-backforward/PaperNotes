# Personalized Question Routing via Heterogeneous Network Embedding

------

## **Motivation**

​	在问答社区（CQA）中保证回答丰富度和社区活跃度的一项关键技术便是问题路由，即针对提出的问题自动识别那些潜在的高质量回答的提供者，从而定向的产生邀请。

​	在之前的工作中提出了很多针对此问题的方法，包括基于特征工程的、基于矩阵填补的、基于额外的社交关系的、基于文本特征的。这些方法均取得了一定的成果，但是先前的工作都没有考虑问题的提出者和回答者在领域及专业背景方向上的匹配程度，即优秀的回答者理论上应当与问题的提出者擅长相近的领域。

​	基于这样的假设，本文提出了一种异构网络嵌入的方法来建模提问者和回答者的表示，同时使用LSTM提取问题的文本特征作为补充，最后将得到的低维特征通过一个CNN来得到用户-问题对的打分，进而排序实现推荐。

## **问题定义**

​	针对获取到的数据，整理为问题集合Q和用户集合U，以用户是否提问和回答为标准进一步划分U为提问者集合R和回答者集合A，R∪A=U。以R∪A∪Q作为顶点，R-Q提出关系、Q-A回答关系为边构建一个异质网络，即三分图。

​	给定这个三分图和一个询问（r，q）需要得到一个集合A的排序，排序越考前则越是潜在的高质量回答者。

## **Model**

![image-20191221113113119](C:\Users\89383\AppData\Roaming\Typora\typora-user-images\image-20191221113113119.png)

​	整个模型可以分为两个部分：Embedding和Scoring 。Embedding部分学习到R、A、Q中每一个entity的低维embedding，Scoring部分利用询问（r，q）的embedding遍历整个A集合，用CNN作为打分函数F，得到三元组的分值![image-20191221114004315](C:\Users\89383\AppData\Roaming\Typora\typora-user-images\image-20191221114004315.png)	下面分别解析这两个部分。

### Embedding

​	首先对于问题节点Q，使用LSTM将变长的问题文本变成一个定长的向量，这一步是NLP里面非常成熟的一个方法了，就不在赘述。

​	针对R、A的表示，使用的基本上也是现成的Graph embedding方法：metapath2vec，这个方法的详细内容可以查看KDD2017上的原始论文，主要就是在图上预先定义一个metapath（此文中推荐使用“A-Q-R-Q-A”和“A-Q-A”），再严格按照此顺序进行random walk，将得到的序列使用Skip-gram的方法训练得到，除了使用LSTM得到Q节点的embedding而非Skip-gram学到，以及目标函数定义为best-answerer和answerer之差加上正负样本之差之外（公式见下图），整个结构和训练过程与原始的metapath2vec一致。

![image-20191221120119292](C:\Users\89383\AppData\Roaming\Typora\typora-user-images\image-20191221120119292.png)

### Scoring		

​	在上面的目标函数中F由CNN实现，将上一步得到的embedding ∈Rd 拼接得到feature矩阵M∈Rd×3，经过三种卷积核k1∈Rd×1，k2∈Rd×2，k3∈Rd×3，得到的feature map拼接经过FC得到score输出，通过两个相同的CNN并行计算得到正样本的score和负样本的score来得到上图中的目标函数。

​	在实际的训练中采取交替优化的方法，skip-gram的无监督目标函数和上图中的Ranking score交替训练，互为正则。

## **实验**

​	作者开源了实验的源代码，注释还是比较清晰，实现random walk、Skip gram类似的方法可以参考。

### baseline

Score：将历史上排第一次数最多的用户作为推荐而不考虑问题

NMF：使用矩阵分解的方法处理共现矩阵进行推荐

L2R：将用户-问题之间的交叉特征送入SVM中来进行排序

### 指标结果

![image-20191221144423433](C:\Users\89383\AppData\Roaming\Typora\typora-user-images\image-20191221144423433.png)

------

## 个人见解

​	这篇文章最大的创新点在于把问题提出者的特征融合进模型，使得得到的用户表示包含了作为回答者和提问者的两种角色关系，同时把LSTM建模的语义信息放到网络中，让embedding不仅包含网络的结构信息又包含语义信息。但是文章中也说到了冷启动问题，当用户出现冷启动时，不论是提问者还是回答者都无法得到一定程度的缓解，这也是将来一个可以升级的地方。关于实验，作者分别选择了两个领域的数据集分别实验，而同一领域提供了一个很强的先验，即问题相似性很强、用户高度关注该领域，所以此方法针对大规模、多领域的数据是否还有很好的效果是一个值得验证的问题。

