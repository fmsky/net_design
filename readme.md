#网络设计日志
***
这部分记录caffe中一些Layer和parameter的说明

####solver.prototxt
* net : 指定配置文件。如train.prototxt
* test_iter :  测试迭代数。如：10000个样本，batch_size为32，则迭代10000/32=313
* test_interval :  每训练迭代test_interval次进行一次测试，例如50000个训练样本，batch_size为64，那么需要50000/64=782次才处理完一次全部训练样本，记作1 epoch。所以test_interval设置为782，即处理完一次所有的训练数据后，才去进行测试。
* base_lr ：基础学习率。学习策略使用的参数。
* momentum ：动量。
* weight_decay：权重衰减。
* lr_policy：学习策略。可选参数：fixed、step、exp、inv、multistep
######lr_policy参数说明：
```
fixed：保持base_lr不变
step：如果设置为step，则需要一个stepsize，返回        
base_lr*gamma^(floor(iter/stepsize)),其中iter表示当前迭代的次数；
exp：返回base_lr*gamma^iter,iter为当前迭代次数
inv：设置inv还需设置一个power，返回base_lr*（1+gamma*iter）^(-power)
multistep：设置为multistep，则还需要设置一个stepvalue，这个参数和step相似，step是均匀等间隔变化，而multistep则是根据stepvalue值变化；
stepvalue参数说明：
  poly：学习率进行多项式误差，返回base_lr*（1-iter/max_iter）^(power)
  sigmoid:学习率进行sigmoid衰减，返回base_lr*（1/(1+exp(-gamma*(iter-stepsize)))）
```
* display：每迭代display次显示结果
* max_iter：最大迭代次数，如果想训练100epoch，则设置max_iter=100*test_interval
* snapshot：保存临时模型的迭代数
* snapshot_prefix：模型前缀
* solver_mode：优化模式。可以使用GPU或者CPU

####Layer
1. Convolution Layer
通俗理解为提取图像的特征，向后传播提取的特征，特征数由num_output定。
```
  layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data" #输入
  top: "conv1"  #输出
  param { 
    lr_mult: 1  #weight的学习率
  }
  param {
    lr_mult: 2  #bias（偏执项）的学习率
  }
  convolution_param {
    num_output: 20 #输出特征提取数
    kernel_size: 5  #卷积核大小为5x5
    weight_filler {
      type: "xavier"  #权值初始化方式
    }
    bias_filler {
      type: "constant" #偏执项的初始化。一般设置为“constant”，值全为0
    }
  }
}
```  
2. Pooling Layer
通过卷积获得特征，利用这些特征做分类，计算量依旧会很大，并且，如果分类器特征输入过多，极易出现过拟合。因此，为了描述大的图像，对不同位置的特征进行聚合统计，可以计算图像一个区域上的某个特定特征的平均值 (或最大值)。这些概要统计特征不仅具有低得多的维度 (相比使用所有提取得到的特征)，同时还会改善结果(不容易过拟合)。这种聚合的操作就叫做池化 (pooling)，也称为平均池化或者最大池化 (取决于计算池化的方法)。
```
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX #最大池化
    kernel_size: 3
    stride: 2
  }
}
```
3. Dropout Layer
定义dropout层，目的是防止cnn过拟合，在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已)。如：训练过程中dropout为0.3，我们在测试时不会进行dropout,而是把输出乘以0.7。
```
 layer {          
  name: "drop7"  
  type: "Dropout"  
  bottom: "fc7"  
  top: "fc7"  
  dropout_param {  
    dropout_ratio: 0.3     #定义选择节点的概率  
  }  
} 
```
4. ReLU Layer
在标准的ReLU激活函数中,当输入为x时,如果x>0,则输出 x,如果输入<=0,则输出0,即输出为max(0,x);
在非标准的ReLU激活函数中,当输入x<=0时, 输出为x * negative_slop(它是一个参数,默认为0)
```
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
```

####梯度下降法
***
以线性回归算法为例：$$h_\theta=\sum_{j=0}^n\theta_jx_j $$
对应的损失函数为：$$J_{train}{(\theta)}=\frac{1}{(2m)\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})}$$
下面是一个二维参数（$\theta_0$和$\theta_1$）组对应损失函数的可视化图：
![alt text](http://images2015.cnblogs.com/blog/764050/201512/764050-20151230183324042-1022081727.png "title")
#####1.批量梯度下降法
批量梯度下降法（Batch Gradient Descent，简称BGD）是梯度下降法最原始的形式，它的具体思路是在更新每一参数时都使用所有的样本来进行更新，其数学形式如下：
    &emsp;(1)对上述的损失函数求偏导：
$$\frac{\delta J(\theta)}{\delta\theta_j}=-\frac{1}{m}\sum_{i=1}^m(y^j-h_\theta(x^i))x_j^i$$
    &emsp;(2)由于是最小化风险函数，所以按照每个参数$\theta$的梯度负方向来更新每个$\theta$:
$$\theta_j^{'}=\theta_j+\frac{1}{m}\sum_{i=1}^m(y^i-h_\theta(x^{(i)}))x_j^i$$
    &emsp;具体伪代码形式为：
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;repeat{
      $$\theta_j^{'}=\theta_j+\frac{1}{m}\sum_{i=1}^m(y^i-h_\theta(x^{(i)}))x_j^i$$
      &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(for every j=0,...,n)
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;}
从上面公式可以注意到，它得到的是一个全局最优解，但是每迭代一步，都要用到训练集所有的数据，如果样本数目m很大，那么可想而知这种方法的迭代速度！所以，这就引入了另外一种方法，随机梯度下降。
    &emsp;**优点：**全局最优解；易于并行实现；
    &emsp;**缺点：**样本数量大时，训练过程缓慢。
从迭代次数上看，BGD迭代的次数相对较少。其迭代的收敛曲线示意图如下：
![alt text](http://images2015.cnblogs.com/blog/764050/201512/764050-20151230190320667-1412088485.png "title")
#####2.随机梯度下降法SGD
由于批量梯度下降法在更新每一个参数时，都需要所有的训练样本，所以训练过程会随着样本数量的加大而变得异常的缓慢。随机梯度下降法（Stochastic Gradient Descent，简称SGD）正是为了解决批量梯度下降法这一弊端而提出的。
　　&emsp;将上面的损失函数写为如下形式：
$$J(\theta)=\frac{1}{m}\sum_{i=1}^m\frac{1}{2}(y^i-h_\theta(x^i))^2=\frac{1}{m}\sum_{i=1}^m\cos t(\theta,(x^i,y^i))$$
$$emsp;&emsp;$\cos t(\theta,(x^i,y^i))=\frac{1}{2}(y^i-h_\theta(x^i))^2$$
    &emsp;利用每隔样本的损失函数对$\theta$求偏导得到对应的梯度，来更新$\theta$：
$$\theta_j^{'}=\theta_j+(y^i-h_\theta(x^i))x_j^i$$
    &emsp;具体的伪代码形式为：
    &emsp;&emsp;1.Randomly shuffle dataset;
    &emsp;&emsp;2.repeat{
    &emsp;&emsp;&emsp;for i=1,...,m{
    &emsp;&emsp;&emsp;&emsp;$\theta_j^{'}=\theta_j+(y^i-h_\theta(x^i))x_j^i$ 
    &emsp;&emsp;&emsp;&emsp;(for j=0,...,n)                     
    &emsp;&emsp;&emsp;}
    &emsp;&emsp;}
随机梯度下降是通过每个样本来迭代更新一次，如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次。但是，SGD伴随的一个问题是噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。
    &emsp;**优点：**训练速度快；
    &emsp;**缺点：**准确度下降，并不是全局最优；不易于并行实现。
从迭代的次数上来看，SGD迭代的次数较多，在解空间的搜索过程看起来很盲目。其迭代的收敛曲线示意图可以表示如下：
![alt text](http://images2015.cnblogs.com/blog/764050/201512/764050-20151230193523495-665207012.png "title")
#####3.小批量梯度下降法MBGD
有上述的两种梯度下降法可以看出，其各自均有优缺点，那么能不能在两种方法的性能之间取得一个折衷呢？即，算法的训练过程比较快，而且也要保证最终参数训练的准确率，而这正是小批量梯度下降法（Mini-batch Gradient Descent，简称MBGD）的初衷。
    &emsp;&emsp;MBGD在每次更新参数时使用b个样本（b一般为10），其具体的伪代码形式为：
    &emsp;&emsp;Say b=10,m=1000.
    &emsp;&emsp;Repart{
      &emsp;&emsp;&emsp;for i=1,11,21,31,...,991{
        &emsp;&emsp;&emsp;&emsp;$\theta_j:=\theta_j-\alpha\frac{1}{10}\sum_{k=i}^{i+9}(h_\theta(x^{(k)})-y^{(k)})x_j^{(k)}$
        &emsp;&emsp;&emsp;&emsp;(for every j=0,...,n)
      &emsp;&emsp;&emsp;}
    &emsp;&emsp;}
#####4.总结
&emsp;**Batch gradient descent:** Use all examples in each iteration；
&emsp;**Stochastic gradient descent:** Use 1 example in each iteration；
&emsp;**Mini-batch gradient descent:** Use b examples in each iteration.
###测试
