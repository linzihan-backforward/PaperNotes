# User Profiling through Deep Multimodal Fusion

------

## Motivation

​	社交媒体上的用户画像一直是一个值得关注的研究热点，在之前的各种不同的用户建模技术中，考虑了多种不同性质用户数据的模型非常有限。考虑到一个用户在社交媒体上会产生各种不同领域的数据，如：文本、图片、关系等，如何建模这些数据之间的联系并使其共同为用户画像提供帮助是一个值得思考的难题。本文提出了一种多模型融合的方法能够将上述的多种数据共同使用来预测用户的性别、年龄、社交兴趣，并取得了非常好的效果。

## Model

​	在本文中我们使用三种不同类型的数据，用户的历史发布状态形成文本数据集，用户的信息图片作为图像数据集，用户与其喜欢的页面形成关系数据集，在融合三种数据之前，我们需要先将其转换为可以计算的特征向量。

​	对于文本数据集，我们使用Linguistic Inquiry and Word Count（LIWC）方法所定义的88个文本特征来得到一个用户的语料库的所有统计信息，具体如何得到可以搜索此方法的相关资料。

​	对于用户的头像图片，我们使用Oxford Face API来捕捉人脸的64个不同的特征，如人脸的各部分位置，发色，性别，眼睛等。

​	对于用户的连接数据，我们将整个图使用无监督的Node2Vec方法，将图中的用户和页面节点统一映射到127维的向量上，选择其中用户节点作为后续使用的特征。

​	在得到不同数据的向量表示之后，下一步就是将这些聚合起来来共同预测用户的属性，这里我们使用一种特殊的DNN作为融合方法，对于上述的三种数据，可以得到7中不同的组合方式，我们的每一个DNN结构都是采用这7中之一的组合作为数据的输入。

![image-20200307120619485](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2018%5D%20User%20Profiling%20through%20Deep%20Multimodal%20Fusion/image-20200307120619485.png?raw=true)

​	可以看到，图中T、I、R、TR等分别代表此NN使用的是哪个数据集合作为输入。图中的另外一个结构便是紫色部分，其作用是保存每一个DNN块的输出，使用这些输出作为下一次的输入来增加7个模块之间的相互协作。

![image-20200307121047478](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2018%5D%20User%20Profiling%20through%20Deep%20Multimodal%20Fusion/image-20200307121047478.png?raw=true)

​	形式化的计算公式如图，其中D代表一个数据集，加号的左边便是直接对所有的输入数据类型加权，即传统的DNN，而等号右边则是对应紫色的输入部分，z代表不同的DNN结构，t代表此数据在上一次经过第z个结构计算出的输出，将每一个结构的输出再乘上对应的系数α，作为新的输入再计算，这样对于一个DNN结构可以利用上其他相同结构的输出，也就变相的实现了不同种数据组合所带来的不同功效。

## Experiments

### 	Datasets

​	**MyPersonality** 数据集，来自于Facebook，包含用户的基本信息和关联Facebook的行为信息，预测的用户个性为5种个性上的得分，包括外向性，亲和性，负责行等。得分为1-5之间的实数

### Baselines

 	对比方法为仅仅使用单一数据内容或者两个数据内容的方法

### Results

![image-20200307142119715](https://github.com/linzihan-backforward/PaperNotes/blob/master/WSDM/%5BWSDM2018%5D%20User%20Profiling%20through%20Deep%20Multimodal%20Fusion/image-20200307142119715.png?raw=true)

## 个人见解

​	这篇文章的任务是user profiling，基本思想是想利用上多种不同性质的数据，读之前以为其是一个统一的端到端模型包含CV和NLP，但是读完有点觉得不对劲了，不仅仅模型上，行文上也是怪怪的，东一扯西一扯，中间甚至有点搞不懂是在干嘛了，读完后面才有点明白，原来模型就是将三种类型的数据分别预处理到向量，然后再枚举不同的组合方式通过DNN？这就完了？那些子集意义何在？一个全部的包含三个数据输入的DNN难道不包含所有的吗，有点云里雾里。实验部分也是有点充数的意思，没有用其他的有力的对比方法。除了这些怪的地方，至少看到一个创新，那就是还能够利用跨epoch的信息传递来设计模型。