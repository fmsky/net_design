#-*- coding:utf-8 -*-
import caffe


def write_solver():
    solver_string = caffe.proto.caffe_pb2.SolverParameter()
    solver_file = "model/solver.prototxt"
    solver_string.train_net = "model/train.prototxt"
    solver_string.test_iter.append(100)
    solver_string.test_interval = 500
    solver_string.base_lr = 0.0001
    solver_string.momentum = 0.9
    solver_string.weight_decay = 0.004
    solver_string.lr_policy = 'fixed'
    solver_string.display = 100
    solver_string.max_iter = 4000
    solver_string.snapshot = 500
    solver_string.snapshot_format = 1
    solver_string.snapshot_prefix = "model/caffe_model"
    solver_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

    with open(solver_file, 'w') as f:
        f.write(str(solver_string))


if __name__ == '__main__':
    write_solver()