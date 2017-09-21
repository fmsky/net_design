#网络设计日志
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