# Efficient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation

------

## Motivation

​	在Top-N推荐的一系列方法中，负采样（Negative Sampling，NS）是一个被广泛用来定义Loss的方法，尽管模型的框架和输入各不相同，只要是期望得到Ranking的方法，往往都会用到负采样配合pair-wise的损失函数，但是其负采样也存在一些问题：如波动性很强，只能使用一种用户反馈。为了解决负采样存在的问题，我们提出了一种能够同时考虑用户的多种反馈（购买、收藏、加购）的协同过滤模型，其不仅仅在多种行为上取得了很好的效果，还极大地提升了训练的效率。

## Model

​	在用户的商品的交互中虽然有多种行为，但是推荐关心的是其中的一种（如购买）所以对于K中用户的不同类型的行为，我们假设其第K个是关心的，而前K-1个为辅助行为，那么与传统的CF类方法一样，可以得到K个0/1矩阵分别代表这K个行为所产生的交互矩阵，并且对于这K个行为，用户和商品共享同样的embedding，当用户和商品进行交互时，其embedding vector在dot之后还要与目标行为再进行dot以区别不同行为下的得分。

![image-20200212174721800](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200212174721800.png?raw=true)

​	在明确目标值的计算方式后，接下来我们需要将用户的不同行为之间建立联系，从而将上式中的行为孤立的h向量之间产生联系。用户在不同的行为之间是会进行转换的，最终的购买行为之前可能会进行浏览、加购等行为，所以其对应的行为向量h应当能够刻画这种转换关系，参考知识图谱中的关系表示，两个行为之间的转移可以通过转移矩阵刻画。

![image-20200212190736114](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200212190736114.png?raw=true)

​	那么对于一个行为，其理论上可以由任何多种关系转移而来，所以将这些转移求和来得到目标行为。

![image-20200212191003209](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200212191003209.png?raw=true)

​	在所有使用的行为中，由前序行为可以向后续行为进行迁移，那么总有一些初始的行为没有前序，对于这些初始行为，可以采用随时初始化的方法得到，之后沿着整个DAG可以计算得到所有的行为对应的向量。

​	在得到了所有的行为向量之后，如何仅仅使用正样本来计算最终的Loss便成为了最后的问题，将整个的CF考虑成一个回归问题，我们需要对矩阵上每一个位置的值求L2距离，然后再加起来，但是在目标矩阵中很多值都是0，所以可以将其非0值和0值分开考虑，将平方项拆开，可以将Loss简化为两部分，一部分仅仅使用那些非0项即正样本计算，另一部分使用全体值计算。

![image-20200212193715259](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200212193715259.png?raw=true)

​	进行遮掩的个简化之后，计算量高的项就变到了后面的全矩阵的计算上，我们使用上面定义的R的计算公式再进一步带入，同时将权重c简化为仅仅与商品相关，则最外层对于所有用户的求和和所有商品的求和可以跟里层对于维度的求和交换位置，之后所有项分批计算既可以将原本O（B*V*d）的复杂度变成O（（B+V）*d^2）实现计算量的简化，在得到对于一个行为的Loss计算之后，多任务学习就是把每一个行为的预测任务统一到一起进行优化，那么最终的Loss便将每一个行为的公式相加。

![image-20200212194455444](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200212194455444.png?raw=true)

## Experiments

### Datasets

- ​	**Movielens-1M**  这是一个只包含单一行为的数据集
- ​    **beibei**  在NMTR论文中用到并开放的数据集
- ​    **taobao** 包含三种行为（浏览，加购，购买）

### Baselines

- **BPR** 一个使用pair-wise方法的最基础模型
- **ExpoMF** 基于全部数据的MF方法，将所有的确实项作为负样本
- **NCF** 使用MLP来加强MF的方法
- **CMF** 同时对多个行为矩阵进行矩阵分解的方法
- **MC-BPR**  修改BPR中的负采样来适应多种行为
- **NMTR** 最新的结合NCF和多任务学习的方法

### Results

![image-20200213093330012](https://github.com/linzihan-backforward/PaperNotes/blob/master/AAAI/%5BAAAI2020%5D%20Efficient%20Heterogeneous%20Collaborative%20Filtering%20without%20Negative%20Sampling%20for%20Recommendation/image-20200213093330012.png?raw=true)

## 个人见解

​	这篇文章整体的motivation的点其实非常平常，就是考虑多种行为之间的转化关系，没有什么创新的地方，模型上也是很简单的一个协同矩阵分解的模型，一点也不复杂，关键就是抛弃负采样这一点，这是模型具体实现上的优化，按理说跟多种行为转化这个初始的创新点没什么关系，相当于就是把两个不同的点合并进了一个工作里面，而且恰恰是后面这个方法上的改进让整篇工作有了一定的深度，关于Loss推导的地方我也不敢说100%看懂，可以看到的是最后的复杂度化简中模型中的点乘发挥了很大的作用，这感觉非常的巧妙，也是让人觉得高大上的地方，使得这类方法跟堆模型的方法上有了显著的区别。