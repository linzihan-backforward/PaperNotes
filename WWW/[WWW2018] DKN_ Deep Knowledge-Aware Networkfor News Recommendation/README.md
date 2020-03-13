# DKN： Deep Knowledge-Aware Network for News Recommendation

------

## Motivation

​	新闻推荐与其他的商品推荐存在一些不同之处。新闻具有高度时效性和话题敏感性的特点，一般而言新闻的热度不会持续太久，而且用户关注的话题也多是有针对性的。其次，新闻的语言高度浓缩，往往包含很多常识知识，而目前基于词汇共现的模型，很难发现这些潜在的知识。因此本文提出了 DKN，将知识图谱的实体表示与基本的词的语义表示结合起来，融合到新闻推荐系统中。

## Model

​	我们的模型分为两步，首先是将一个新闻标题表示为一个向量，然后利用用户的所有历史向量，得到用户的表示向量，最后二者交互得到最终的得分。

### Knowledge Distillation

​	第一步是从标题中抽取出一些关键词并将其与知识图谱中的实体进行对齐，利用知识图谱嵌入的方法得到这些关键词的向量表示

![image-20200311172314064](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311172314064.png?raw=true)

​	这一步用到的主要是实体连接技术和知识图谱嵌入技术，如TransE、TransD、TransR等，这样对于新闻标题中的一个词，其存在两个向量表示，一个是使用传统word2vec技术得到的语义表示，一个是其对应实体在知识图谱中的知识表示。有了这两个是不是就可以得到新闻的表示了呢，在实践中，发现这样依然不能很好的包含知识实体的关系，所以我们将所有的上述实体在知识图谱中的直接邻居拿过来作为此新闻的上下文实体一并考虑，一个实体的上下文定义为在KG中与其直接连接的其他实体

![image-20200311172829146](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311172829146.png?raw=true)

​	这样，对于一个新闻，我们可以从中解析出三部分的内容共同建模新闻的表示：词向量、实体向量、上下文实体向量。接下来使用CNN的方法来计算这三部分所构成的矩阵。

### Knowledge-aware CNN

​	在得到上面三种向量之后，一种直接的办法是将所有的词向量作为一个序列整体送入Text-CNN中，但是这种方法将三部分的关系忽略了，词与实体之间存在对应关系同时其向量位于不同的向量空间中，采取完全相同的连接操作会导致空间的混乱，最后连接操作还要求其具有相同的维度，这会增加额外的限制。为了保留三部分的关系克服上述的局限，我们采用一种词-实体对齐的KCNN来得到新闻整体的向量。

![image-20200311184639348](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311184639348.png?raw=true)

​	我们将上述三部分向量作为三个独立的通道，存在对应关系的词和实体在序列的同一位置实现对齐，同时使用变换函数g来将实体表示映射到语义空间。之后，对于d×n×3的输入矩阵，我们使用Text-CNN模型来获取其特征值。

![image-20200311184900725](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311184900725.png?raw=true)

![image-20200311184907660](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311184907660.png?raw=true)

​	使用m个不同大小的卷积核就可以将输入矩阵转换为一个m维的向量。

![image-20200311185115299](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200311185115299.png?raw=true)

### Attention-based User Interest Extraction

​	给定一个用户的所有新闻点击历史以及其对应的新闻向量表示，如何聚合得到用户的表示呢，考虑到用户多角度的兴趣，以及当前备选新闻的影响，我们使用当前备选新闻与历史点击新闻进行Attention，再加权聚合所有历史表示

![image-20200312092430667](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200312092430667.png?raw=true)

利用得到的s权值加权聚合所有的e，打分时将用户表示和新闻表示一同送入DNN打分器即可

![image-20200312092545725](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200312092545725.png?raw=true)

## Experiments

### Datasets

​	数据集来自于Bing News，知识图谱为Microsoft Satori knowledge Graph中的一个子图，包含所有新闻中出现的实体和它们的一阶邻居

### Baselines

- **libFM**

- **KPCNN**

- **DSSM**

- **DeepWide**

- **DeepFM**

- **YouTubeNet**

- **DMF**

  

### Results

![image-20200312094444308](https://github.com/linzihan-backforward/PaperNotes/blob/master/WWW/%5BWWW2018%5D%20DKN_%20Deep%20Knowledge-Aware%20Networkfor%20News%20Recommendation/image-20200312094444308.png?raw=true)

## 个人见解

​	这篇文章的问题和方向与目前我想做的东西非常相近，新闻推荐的要点是对新闻标题的建模，纯粹使用文本无法解读出新闻之间所包括的常识知识，所以本方法使用一个对齐的知识图谱来弥补这方面的不足，原本的文本依然建模语义信息，而知识图谱中的实体则建模这些知识，二者合并起来形成一个新闻的表示，这样看似是融入了知识信息，实际上模型依旧是两个部分，知识图谱独立得到实体，文本独立得到关键词，然后最后再把二者结合，这样确实是一种简单有效地方法，但是还是没有解决本质问题即从文本中解读出知识，这样的方法就是增加了另一个模型的输入，严重依赖于实体链接和知识图谱补全技术，当新闻中词无法与实体对齐时，这样的方法便完全失效，甚至不能缓解。