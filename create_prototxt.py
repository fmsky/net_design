#-*- coding:utf-8 -*-
import caffe
train_lmdb = "data/img_trian_lmdb"


def create_net(lmdb, mean_file, batch_size, include_acc=False):
    #网络规范
    net = caffe.NetSpec()

    #第一层：Data层
    net.data, net.label = caffe.layers.Data(
        source=train_lmdb,
        backend=caffe.params.Data.LMDB,
        batch_size=64,
        ntop=2,
        transform_param=dict(crop_size=40, mean_file=mean_file, mirror=True))

    #第二层：卷积层
    net.conv1 = caffe.layers.Convolution(
        net.data,
        num_output=20,
        kernel_size=5,
        weight_filler={"type": "xavier"},
        bias_filler={"type": "constant"})

    #第三层：ReLU
    net.relu1 = caffe.layers.ReLU(
        net.conv1, in_place=True)  #in_palce：原地算法，输出覆盖输入

    #第四层：Pooling层
    net.pool1 = caffe.layers.Pooling(
        net.relu1, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)

    net.conv2 = caffe.layers.Convolution(
        net.pool1,
        kernel_size=3,
        stride=1,
        num_output=32,
        pad=1,
        weight_filler=dict(type='xavier'))

    net.relu2 = caffe.layers.ReLU(net.conv2, in_place=True)

    net.pool2 = caffe.layers.Pooling(
        net.relu2, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)

    #全连接fc
    net.fc1 = caffe.layers.InnerProduct(
        net.pool2, num_output=1024, weight_filler=dict(type='xavier'))
    net.relu3 = caffe.layers.ReLU(net.fc1, in_place=True)

    #dropout层:有助于防止过拟合
    net.drop1 = caffe.layers.Dropout(net.relu3, in_place=True)
    net.fc2 = caffe.layers.InnerProduct(
        net.drop1, num_output=10, weight_filler=dict(type='xavier'))

    #softmax层：SoftmaxWithLoss输出为求得样本的loss之和除以样本数的值,Softmax输出为每个类别的概率值
    net.loss = caffe.layers.SoftmaxWithLoss(net.fc2, net.label)

    #训练的prototxt文件不包括Accuracy层，测试的时候需要
    if include_acc:
        net.acc = caffe.layers.Accuracy(net.fc2, net.label)
        return str(net.to_proto)
    return str(net.to_proto())


def write_net():
    global train_lmdb
    mean_file = "model/mean.binaryproto"
    train_proto = "model/train.prototxt"

    #写入train.prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, mean_file, batch_size=64)))


if __name__ == '__main__':
    write_net()