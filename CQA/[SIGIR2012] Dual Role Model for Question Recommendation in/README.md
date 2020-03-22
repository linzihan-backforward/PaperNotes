# Dual Role Model for Question Recommendation in Community Question Answering

------

## Motivation

​	在CQA问题中，不论是为一个问题推荐Top N用户还是为一个用户推荐合适的问题，其本质上都是用户和问题之间的匹配，也就是解决如何表示一个用户的兴趣以及问题。基于这样的核心，很多方法使用隐藏语义模型（PLSA、LDA）来将用户和问题建模为在多个主题上的一个分布。但是我们发现这样的问答社区与其他的推荐场景的核心场景在于：用户即可以作为回答者又可以作为提出者，一个用户在充当这两种角色时其表示应当是不同的，已有的方法并没有考虑这样的不同。所以，在本文中我们提出双角色模型（DRM）来得到两种角色之间的区别和联系并与问题推荐统一到一个概率模型中。

## Model

​	首先，我们定义问题推荐如下：给定问题集合Q，用户集合U，其中Q中每一个问题是一个三元组<t，ua，uq> t代表问题的文本内容，ua代表这个问题的回答者id，uq代表这个问题的提出者id，当一个问题有多个回答者时，将其拆分为多个三元组，当没有回答者时ua为空。整体的模型分为两部分：得到用户和问题的表示，将候选用户按照给定方法排序。得到表示的模块使用的是基于PLSA的模型，根据对用户之间独立性假设的不同，可以分为独立模型和非独立模型。

### 独立模型	

​	![image-20200321144015452](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321144015452.png?raw=true)

​	根据PLSA的建模过程，首先从所有的隐藏主题中选择一个主题z，然后主题z会根据概率产生一个问题q，并根据概率生成一个词w，这样问题q与词w的联合概率表示如下：
$$
P（q,w） = \sum_{z} {P(z)P(q|z)P(w|z)}
$$
​	然后对于所有问题和内容，我们都将其拆分为问题和词的联合概率，生成全部问题的概率的对数可以表示如下：
$$
L = \sum_{q,w}{c(q,w)\log P(q,w)}
$$
​	其中c代表（q，w）出现的频率，最大化这个概率L即是整个PLSA模型的求解过程，这里使用的是EM算法，即E和M交替进行直到概率值收敛的算法：
$$
E-Step：  P（z|q,w） = \frac{P(z)P(q|z)P(w|z)}{\sum_{z'}{P(z')P(q|z')P(w|z')}}
$$

$$
M-Step: P(z) \propto \sum_{q,w}{c(q,w)P(z|q,w)} \\
        P(q|z) \propto \sum_{w}{c(q,w)P(z|q,w)} \\
        P(w|z) \propto \sum_{q}{c(q,w)P(z|q,w)}
$$

​	在计算完所有问题的主题分布之后，一个用户作为回答者和提出者的分布可以分别由其回答的问题和提出的问题求得

![image-20200321152152998](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321152152998.png?raw=true)

### 	依赖模型

​	依赖模型的假设是用户的提问角色和回答角色之间是有影响的，则生成一个问题的概率分布与用户的角色有关：
$$
P(t,u^a,u^q) = \sum_{z}{P(z)P(u^a|z)P(u^q|z)\prod_{w\in t}{P(w|z)^{c(w,t)}}}
$$
​	这样的话，对用户角色的概率估计就与问题的估计交织在一起，相应的EM算法也变得复杂一些

![image-20200321153211843](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321153211843.png?raw=true)

​	当所有的分布参数收敛之后，我们就可以通过计算P(ui|q)，然后将其最大的Top  N个用户作为面向问题q的推荐

![image-20200321153600310](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321153600310.png?raw=true)

## Experiments

### 	Datasets

​	从Yahoo！Answers网站中爬取的问题和用户，根据用户回答或者提问的数量，分为了三个稀疏程度不同的数据集

![image-20200321154415216](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321154415216.png?raw=true)

### Baselines

- **VSM** 将用户和问题表示为一个|W|维度的tfidf权重向量，余弦距离衡量相似度
- **PLSA** 一个单角色的隐藏因子模型

###  Results

![image-20200321155909853](https://github.com/linzihan-backforward/PaperNotes/blob/master/CQA/%5BSIGIR2012%5D%20Dual%20Role%20Model%20for%20Question%20Recommendation%20in/image-20200321155909853.png?raw=true)

## 个人见解

​	本文是使用隐藏概率模型来解决CQA问题的一篇工作，其主要的点在于用户作为回答者和提问者时在不同主题上面的分布是具有很大区别的，这一点我感觉并不是让人特别想不到的那种novelty的想法，但是本文将这样的一个想法与PLSA这种那是非常常见的概率模型相结合，除了两种不同的假设所带来的推导方式上的不同之外，还可视化的通过KL散度展示出两种分布的不同，时刻在反馈开头时提出的idea，给读者一种详尽充实的感觉。我感觉对于EM算法的推导上面理解的还是比较吃力，但是能感觉到这篇文章还是能配得上SIGIR的。