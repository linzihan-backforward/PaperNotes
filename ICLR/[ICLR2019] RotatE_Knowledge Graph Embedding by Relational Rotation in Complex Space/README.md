# RotatE：Knowledge Graph Embedding by Relational Rotation in Complex Space

------

## Motivation

​	在知识图谱的补全任务中，影响模型表现的一个关键点便是对关系模式的解读，知识图谱中的关系多种多样，根据其连接的实体可以大致形成 对称关系、非对称关系、反转关系、包含关系等大类。基于这样的发现，我们提出了一个知识图谱嵌入的方法，将关系抽象为复数向量空间的一种旋转，以此来高效的区分不同类型的关系，最终在补全任务上取得SOTA效果。

## Model

​	在知识图谱嵌入任务中，不同的模型所重点关注的关系类别是不同的，而多种关系有共存于整个的知识图谱中，所以如果让模型尽可能地适应多种的关系就成为了保证模型性能的关键。在介绍模型之前，首先对知识图谱中三种最典型的关系对称、反转、组成，进行形式化的定义。

![image-20200226105509463](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20RotatE_Knowledge%20Graph%20Embedding%20by%20Relational%20Rotation%20in%20Complex%20Space/image-20200226105509463.png?raw=true)

​	对于我们的模型，我们定义一个三元组关系应当满足如下的形式

![image-20200226110122749](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20RotatE_Knowledge%20Graph%20Embedding%20by%20Relational%20Rotation%20in%20Complex%20Space/image-20200226110122749.png?raw=true)

​	其中⭕代表element-wise的乘积，r中的每一个元素ri为一个复数，代表在复平面上逆时针旋转θ的角度，即ri为一个模长等于角度可变的复数，通过这样的运算，上面的乘积关系实际上是对，h向量中的每一个值在复数平面进行了一个旋转，这样得到的t同样在复数空间，我们可以用复平面上的欧氏距离定义二者的距离

![image-20200226110741132](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20RotatE_Knowledge%20Graph%20Embedding%20by%20Relational%20Rotation%20in%20Complex%20Space/image-20200226110741132.png?raw=true)

​	通过将实体表示到复平面上的向量，而关系对应为旋转，上面定义的三种不同类型的关系其实都能够涵盖在模型可处理的范围内

![image-20200226111006317](C:\Users\89383\AppData\Roaming\Typora\typora-user-images\image-20200226111006317.png)

​	我们的模型于TransE方法在本质上的不同在于将原本实数空间的运算拓展到了复数空间，利用旋转的可重复性来解决了TransE模型所不能解决的对称问题。于其他的嵌入方法一样，训练的目标函数定义为基于负采样的pair-wise Loss

![image-20200226111455404](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20RotatE_Knowledge%20Graph%20Embedding%20by%20Relational%20Rotation%20in%20Complex%20Space/image-20200226111455404.png?raw=true)

## Experiments

### Datasets

​	四个常用的知识图谱数据集：**FB15k** **WN18** **FB15k-237** **WN18RR**

### Baselines

​	除了常用的TransE、DistMultE、ConvE等之外，我们将我们提出的RotatE模型中实体向量每一维度的模长固定，作为一个变种模型baseline pRotatE

### Results

![image-20200226113930482](https://github.com/linzihan-backforward/PaperNotes/blob/master/ICLR/%5BICLR2019%5D%20RotatE_Knowledge%20Graph%20Embedding%20by%20Relational%20Rotation%20in%20Complex%20Space/image-20200226113930482.png?raw=true)

## 个人见解

​	本文是在ICLR上为数不多的偏应用类的知识图谱文章，理论性相对较浅，实验性更多，但是读下来感觉模型背后的数学思想还是非常深厚的。复数域是一个数学界习以为常的研究角度，但是计算机算法中大多还是喜欢在实数域处理问题，这也使得我们在思想上认为只有实数可以用来解决问题，这篇文章让知识图谱领域打开了思路，通过将所有的对象和运算定义在复数域，我们发现之前棘手的问题迎刃而解，我觉得正是这种思想上的开创性让其能够入选ICLR这个理论会议

