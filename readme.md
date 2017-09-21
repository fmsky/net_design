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
