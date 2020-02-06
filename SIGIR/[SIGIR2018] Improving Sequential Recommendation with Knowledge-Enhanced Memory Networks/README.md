# Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks

------

## Motivation

​	序列推荐问题将每一个用户的行为按照时间排成序列，使用已有的序列来预测下一个交互的商品，使用序列化的神经网络（RNN）能够从行为序列中学习到用户的偏好，但是无法很好地建模出用户细粒度的属性级别的喜好，同时这种神经网络产生的向量没有可解释性。为了解决这两点，我们通过引入外部的知识的形式加强对偏好的建模，考虑到记忆网络能够很好的储存大量个性化的键值对，我们使用一个知识图谱加强的记忆网络来建模属性级别喜好，同时与原本的RNN相结合来吸收二者的优点。

## Model

​	我们的模型包含两个部分，一个使用RNN的模型来刻画用户的序列偏好，一个使用知识附加的记忆网络来刻画用户在属性级别的偏好，最后将两个部分的输出综合作为用户的低维表示，下面分别介绍这两个部分。

### A GRU-based Sequential Recommender

​	对于一个用户的交互行为序列，将其交互的商品按照时间排序后可以形成一个item序列，而GRU在处理这样的时间序列上具有突出的效果，所以我们将{i1……it}序列依次输入GRU来得到最终的一个向量表示，并将其叫做用户的序列偏好表示。

![image-20200205170520016](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200205170520016.png?raw=true)

​	其中q表示商品的嵌入，可以是预训练好之后固定的，也可以是模型的参数。

### Augmenting Sequential Recommender with Knowledge-Enhanced Memory Networks

​	下面介绍一个全新的包含额外知识的键值对记忆网络模型。这个模型在一定程度上与上面的GRU序列模型相独立，用于刻画细粒度的属性上的偏好。首先一个公认的点是对于一个领域的推荐item，有一系列相关的属性于其相关（如推荐电影则要考虑：演员、导演、类型等属性），我们希望通过一个额外的存储结构来细粒度的刻画每一个用户在这些方面的偏好，这样的目的可以通过一个键值对记忆网络来实现。

​	具体地，这个记忆网络（又成KV-MNS）包括每一个用户相独立的一个向量对集合，每一个集合包含A个向量对，每一个对包含一个键向量代表一个属性，值向量代表本用户在此属性上的偏好，注意键向量对于所有的用户来说是相同的，因为数据集中包含的属性是一样的，而值向量是不同的，因为每一个用户的偏好是不同的。这个存储结构应该支持读操作来取出一个用户各方面的偏好，同时支持写属性来对用户的偏好进行更新。

​	读写操作都需要一个query的输入，读操作根据此输入来输出对应的结构，而写操作则将此输入写入存储网络中，这个query最好的选择便是序列推荐部分得到的用户序列偏好表示ht。![image-20200205191923848](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200205191923848.png?raw=true)为了其能够直接参与运算而不用考虑作用空间问题，使用MLP进行映射

![image-20200205191901688](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200205191901688.png?raw=true)

一个抽象的读操作可以表示如上图。其输出mt则表示用户u在时间t在属性上的偏好表示。后面会介绍其具体地实现。写操作所对应的输入与读操作不同，当用户与商品交互后，我们希望把这个商品对应的属性写入记忆里，所以其输出应当为商品的嵌入表示。

![image-20200205192417583](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200205192417583.png?raw=true)

最后需要注意的是，所有的属性向量即键向量是预先设定好的并不是随着训练更新的。

​	在定义后基本的结构之后，我们需要考虑如何将知识图谱融合进上面的结构中。首先，对于推荐商品我们假定其在知识图谱中存在相对应的实体，并且我们可以实现找到这种对应关系，则对于一个目标商品i，可以在知识图谱中找到包含它的三元组（eh，et，r），其中关系r可以理解为实体的一种属性（如实体《阿凡达》的导演为卡梅隆）则这个关系r就可以作为上面说的事先定义的键向量，所以我们通过这种方式来将知识图谱引入记忆网络中。![image-20200206094740197](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206094740197.png?raw=true)

​	现在，我们介绍读写函数具体地实现方法。读函数是根据查询取出对应的值，所以可以使用Attention的方式计算在每一个属性上的权重，然后加权取出。

![image-20200206095047965](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206095047965.png?raw=true)

​	对于写操作则要稍微复杂一点，因为我们希望在值向量中存储每一个用户在每一个属性上的偏好值，所以对于一个新的输入商品自然要从商品中解析出其各个属性值，这时我们的知识图谱又派上了用场，因为我们定义的属性就是知识图谱中的关系，所以使用实体+关系便应当能得到得到对应的取值，这与TransE的思想是一致的，所以对于输入的商品i，我们使用ei+ra来作为其在属性a上的修改值，将其记为eia，采用一种门的概念来计算修改比例，整体的修改方式如下：

![image-20200206100027109](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206100027109.png?raw=true)

​	在两个部分介绍完成之后，需要把两部分的输出进行综合，对于一个输入首先使用序列的模型得到序列偏好表示，然后将其作为查询来对记忆进行读操作，得到属性偏好表示，两部分相连得到最终的用户表示。在商品侧使用商品表示和知识图谱中对应的实体嵌入作为商品表示，二者在使用MLP嵌入后点积作为最终的推荐得分。

![image-20200206100846733](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206100846733.png?raw=true)

​	在训练部分，使用BPR预训练商品嵌入，并在后面的模型训练中固定。同时使用到的实体和关系嵌入同样固定，目标函数使用负采样的pair-loss

![image-20200206101522601](https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206101522601.png?raw=true)

## Experiments

### Datasets

​	对于知识图谱，使用的是Free-BASE数据集，包含630万三元组。

​	对于推荐系统，使用了四个数据集：Last.FM music、MovieLens-20m、MovieLens-1m、Amazon book

### Baselines

- ​	**BPR**  使用pairwise-loss直接优化隐藏因子
- ​    **NCF**  使用NN代替MF中的点乘得到的模型
-    **CKE**  使用知识图谱来加强推荐表现
-    **FPMC** MC和MF的融合模型，用于序列推荐
-    **RUM**  使用额外的记忆空间来加强序列推荐
-    **GRU**  使用GRU来进行序列推荐的方法
-    **GRU++**  在上面模型的基础上使用BPR预训练商品嵌入
-    **GRUF**  使用额外的特征向量与商品嵌入一起通过GRU进行序列推荐

### Results

<img src="https://github.com/linzihan-backforward/PaperNotes/blob/master/SIGIR/%5BSIGIR2018%5D%20Improving%20Sequential%20Recommendation%20with%20Knowledge-Enhanced%20Memory%20Networks/image-20200206123343331.png?raw=true" alt="image-20200206123343331" style="zoom: 67%;" />

## 个人见解

​	这篇文章在完成时正值知识图谱和记忆网络大火的时候，所以这篇文章使用了一种新颖的角度将这二者与推荐系统结合了起来，将关系对应到属性类别，将实体+关系对应到具体的属性值上，这种对应确实不是很容易找到的，同时这种方式不仅仅提供了一种计算用户表示的新方法，而且也同时提供了一种解释性的角度，是一种与之前甚至之后的工作完全不同的角度，在这里序列的GRU模型充当的是一个配角的角色，稍微改造之后，这种方法应该不仅仅可以用于序列推荐，而应该变成广义的推荐方法。同时这种利用对齐实体及相应关系的方法为其他希望引入知识图谱的工作提供了一种启发，但是也会带来一些副作用，如无法对齐的item该怎么处理。