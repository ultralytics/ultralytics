#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import traceback
from Caffe import caffe_net
import torch.nn.functional as F
from torch.autograd import Variable
from Caffe import layer_param
from torch.nn.modules.utils import _pair
import numpy as np
import math
from torch.nn.modules.utils import _list_with_default

"""
How to support a new layer type:
 layer_name=log.add_layer(layer_type_name)
 top_blobs=log.add_blobs(<output of that layer>)
 layer=caffe_net.Layer_param(xxx)
 <set layer parameters>
 [<layer.add_data(*datas)>]
 log.cnet.add_layer(layer)

Please MUTE the inplace operations to avoid not find in graph
"""


# TODO: support the inplace output of the layers

class Blob_LOG():
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


NET_INITTED = False


# 转换原理解析：通过记录
class TransLog(object):
    def __init__(self):
        """
        doing init() with inputs Variable before using it
        """
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.cnet = caffe_net.Caffemodel('')
        self.debug = True

    def init(self, inputs):
        """
        :param inputs: is a list of input variables
        """
        self.add_blobs(inputs)

    def add_layer(self, name='layer'):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        self.detail_layers[name] += 1
        name = '{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for blob in blobs:
            self._blobs_data.append(blob)  # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1
            if with_num:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))
            if self.debug:
                print("{}:{} was added to blobs".format(blob_id, rst[-1]))
            print('Add blob {} : {}'.format(rst[-1].center(21), blob.size()))
            self._blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var):
        var = id(var)
        if self.debug:
            print("{}:{} getting".format(var, self._blobs[var]))
        try:
            return self._blobs[var]
        except:
            print("WARNING: CANNOT FOUND blob {}".format(var))
            return None


log = TransLog()

layer_names = {}


def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    name = log.add_layer(name='conv')
    log.add_blobs([x], name='conv_blob')
    layer = caffe_net.Layer_param(name=name, type='Convolution',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None, groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        #print('conv2d weight, bias: ',weight.cpu().data.numpy(), bias.cpu().data.numpy())

    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    name = log.add_layer(name='conv_transpose')
    log.add_blobs([x], name='conv_transpose_blob')
    layer = caffe_net.Layer_param(name=name, type='Deconvolution',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _linear(raw, input, weight, bias=None):
    x = raw(input, weight, bias)
    layer_name = log.add_layer(name='fc')
    top_blobs = log.add_blobs([x], name='fc_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='InnerProduct',
                                  bottom=[log.blobs(input)], top=top_blobs)
    layer.fc_param(x.size()[1], has_bias=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _split(raw, tensor, split_size, dim=0):
    # split in pytorch is slice in caffe
    x = raw(tensor, split_size, dim)
    layer_name = log.add_layer('split')
    top_blobs = log.add_blobs(x, name='split_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Slice',
                                  bottom=[log.blobs(tensor)], top=top_blobs)
    slice_num = int(np.floor(tensor.size()[dim] / split_size))
    slice_param = caffe_net.pb.SliceParameter(axis=dim, slice_point=[split_size * i for i in range(1, slice_num)])
    layer.param.slice_param.CopyFrom(slice_param)
    log.cnet.add_layer(layer)
    return x


def _pool(type, raw, input, x, kernel_size, stride, padding, ceil_mode):
    # TODO dilation,ceil_mode,return indices
    layer_name = log.add_layer(name='{}_pool'.format(type))
    top_blobs = log.add_blobs([x], name='{}_pool_blob'.format(type))
    layer = caffe_net.Layer_param(name=layer_name, type='Pooling', bottom=[log.blobs(input)], top=top_blobs)

    # TODO w,h different kernel, stride and padding
    # processing ceil mode
    layer.pool_param(kernel_size=kernel_size, stride=kernel_size if stride is None else stride,
                     pad=padding, type=type.upper())
    log.cnet.add_layer(layer)
    if ceil_mode == False and stride is not None:
        oheight = (input.size()[2] - _pair(kernel_size)[0] + 2 * _pair(padding)[0]) % (_pair(stride)[0])
        owidth = (input.size()[3] - _pair(kernel_size)[1] + 2 * _pair(padding)[1]) % (_pair(stride)[1])
        if oheight != 0 or owidth != 0:
            caffe_out = raw(input, kernel_size, stride, padding, ceil_mode=False)
            print("WARNING: the output shape miss match at {}: "

                  "input {} output---Pytorch:{}---Caffe:{}\n"
                  "This is caused by the different implementation that ceil mode in caffe and the floor mode in pytorch.\n"
                  "You can add the clip layer in caffe prototxt manually if shape mismatch error is caused in caffe. ".format(
                layer_name, input.size(), x.size(), caffe_out.size()))


def _max_pool2d(raw, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    x = raw(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    _pool('max', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _avg_pool2d(raw, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    _pool('ave', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _max(raw, *args):
    x = raw(*args)
    if len(args) == 1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        bottom_blobs = []
        for arg in args:
            bottom_blobs.append(log.blobs(arg))
        layer_name = log.add_layer(name='max')
        top_blobs = log.add_blobs([x], name='max_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=bottom_blobs, top=top_blobs)
        layer.param.eltwise_param.operation = 2
        log.cnet.add_layer(layer)
    return x


def _cat(raw, inputs, dimension=0):
    x = raw(inputs, dimension)
    bottom_blobs = []
    for input in inputs:
        bottom_blobs.append(log.blobs(input))
    layer_name = log.add_layer(name='cat')
    top_blobs = log.add_blobs([x], name='cat_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Concat',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.concat_param.axis = dimension
    log.cnet.add_layer(layer)
    return x


def _dropout(raw, input, p=0.5, training=False, inplace=False):
    x = raw(input, p, training, inplace)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='dropout')
    top_blobs = log.add_blobs([x], name=bottom_blobs[0], with_num=False)
    layer = caffe_net.Layer_param(name=layer_name, type='Dropout',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([caffe_net.pb.NetStateRule(phase=0)])  # 1 for test, 0 for train
    log.cnet.add_layer(layer)
    return x


def _threshold(raw, input, threshold, value, inplace=False):
    # for threshold or relu
    if threshold == 0 and value == 0:
        x = raw(input, threshold, value, inplace)
        bottom_blobs = [log.blobs(input)]
        name = log.add_layer(name='relu')
        log.add_blobs([x], name='relu_blob')
        layer = caffe_net.Layer_param(name=name, type='ReLU',
                                      bottom=bottom_blobs, top=[log.blobs(x)])
        log.cnet.add_layer(layer)
        return x
    if value != 0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x = raw(input, input, threshold, value, inplace)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='threshold')
    top_blobs = log.add_blobs([x], name='threshold_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Threshold',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.threshold_param.threshold = threshold
    log.cnet.add_layer(layer)
    return x


def _relu(raw, input, inplace=False):
    # for threshold or prelu
    x = raw(input, False)
    name = log.add_layer(name='relu')
    log.add_blobs([x], name='relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _prelu(raw, input, weight):
    # for threshold or prelu
    x = raw(input, weight)
    bottom_blobs = [log.blobs(input)]
    name = log.add_layer(name='prelu')
    log.add_blobs([x], name='prelu_blob')
    layer = caffe_net.Layer_param(name=name, type='PReLU',
                                  bottom=bottom_blobs, top=[log.blobs(x)])
    if weight.size()[0] == 1:
        layer.param.prelu_param.channel_shared = True
        layer.add_data(weight.cpu().data.numpy()[0])
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope)
    name = log.add_layer(name='leaky_relu')
    log.add_blobs([x], name='leaky_relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.param.relu_param.negative_slope = negative_slope
    log.cnet.add_layer(layer)
    return x


def _tanh(raw, input):
    # for tanh activation
    x = raw(input)
    name = log.add_layer(name='tanh')
    log.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name, type='TanH',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _softmax(raw, input, dim=None, _stacklevel=3):
    # for F.softmax
    x = raw(input, dim=dim)
    if dim is None:
        dim = F._get_softmax_dim('softmax', input.dim(), _stacklevel)
    bottom_blobs = [log.blobs(input)]
    name = log.add_layer(name='softmax')
    log.add_blobs([x], name='softmax_blob')
    layer = caffe_net.Layer_param(name=name, type='Softmax',
                                  bottom=bottom_blobs, top=[log.blobs(x)])
    layer.param.softmax_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _sigmoid(raw, input):
    # for tanh activation
    x = raw(input)
    name = log.add_layer(name='Sigmoid')
    log.add_blobs([x], name='Sigmoid_blob')
    layer = caffe_net.Layer_param(name=name, type='Sigmoid',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _batch_norm(raw, input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation, we first save the parameters

    x = raw(input, running_mean, running_var, weight, bias,
            training, momentum, eps)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='batch_norm')
    top_blobs = log.add_blobs([x], name='batch_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
        #print('running_mean: ',running_mean_clone.cpu().numpy())
        #print('running_var: ',running_var_clone.cpu().numpy())
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
        #print('scale weight: ', weight.cpu().data.numpy())
        #print('scale bias: ', bias.cpu().data.numpy())
    return x


def _instance_norm(raw, input, running_mean=None, running_var=None, weight=None,
                   bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    # TODO: the batch size!=1 view operations
    print("WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1")
    if running_var is not None or weight is not None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError("not implement the affine=True or track_running_stats=True case InstanceNorm")
    x = torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        use_input_stats, momentum, eps, torch.backends.cudnn.enabled)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='instance_norm')
    top_blobs = log.add_blobs([x], name='instance_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
        running_mean = torch.zeros(input.size()[1])
        running_var = torch.ones(input.size()[1])
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
    running_mean_clone = running_mean.clone()
    running_var_clone = running_var.clone()
    layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x


# upsample layer
def _interpolate(raw, input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    # 定义的参数包括 scale,即输出与输入的尺寸比例,如 2;scale_h、scale_w,
    # 同 scale,分别为 h、w 方向上的尺寸比例;pad_out_h、pad_out_w,仅在 scale 为 2 时
    # 有用,对输出进行额外 padding 在 h、w 方向上的数值;upsample_h、upsample_w,输
    # 出图像尺寸的数值。在 Upsample 的相关代码中,推荐仅仅使用 upsample_h、
    # upsample_w 准确定义 Upsample 层的输出尺寸,其他所有的参数都不推荐继续使用。
    '''
    if mode == 'bilinear':      
        x = raw(input, size, scale_factor, mode)
        name = log.add_layer(name='conv_transpose')
        log.add_blobs([x], name='conv_transpose_blob')
        layer = caffe_net.Layer_param(name=name, type='Deconvolution',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
        print('Deconv: ', name)
        print(input.shape)
        print(x.size())
        print(size)
        factor = float(size[0]) / input.shape[2]
        C = x.size()[1]
        print(factor,C) 
        kernel_size = int(2 * factor - factor % 2)
        stride = int(factor)
        num_output = C
        group = C
        pad = math.ceil((factor-1) / 2.)
        print('kernel_size, stride, num_output, group, pad')
        print(kernel_size, stride, num_output, group, pad)
        layer.conv_param(num_output, kernel_size, stride=stride,
                     pad=pad, weight_filler_type='bilinear', bias_term=False, groups=group)

        layer.param.convolution_param.bias_term = False
        log.cnet.add_layer(layer)
        return x
    '''
    # transfer bilinear align_corners=True to caffe-interp
    if mode == "bilinear" and align_corners == True:
        x = raw(input, size, scale_factor, mode)
        name = log.add_layer(name='interp')
        log.add_blobs([x], name='interp_blob')
        layer = caffe_net.Layer_param(name=name, type='Interp',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
        layer.interp_param(size=size, scale_factor=scale_factor)
        log.cnet.add_layer(layer)
        return x

    # for nearest _interpolate
    if mode != "nearest" or align_corners != None:
        raise NotImplementedError("not implement F.interpolate totoaly")
    x = raw(input, size, scale_factor, mode)
    layer_name = log.add_layer(name='upsample')
    top_blobs = log.add_blobs([x], name='upsample_blob'.format(type))
    layer = caffe_net.Layer_param(name=layer_name, type='Upsample',
                                  bottom=[log.blobs(input)], top=top_blobs)
    #layer.upsample_param(size=(input.size(2), input.size(3)), scale_factor=scale_factor)
    #layer.upsample_param(size=size, scale_factor=scale_factor)
    layer.upsample_param(size=None, scale_factor=size[0])
    
    log.cnet.add_layer(layer)
    return x


# ----- for Variable operations --------

def _view(input, *args):
    x = raw_view(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='view')
    top_blobs = log.add_blobs([x], name='view_blob')

    # print('*'*60)
    # print('input={}'.format(input))
    # print('layer_name={}'.format(layer_name))
    # print('top_blobs={}'.format(top_blobs))

    layer = caffe_net.Layer_param(name=layer_name, type='Reshape', bottom=[log.blobs(input)], top=top_blobs)
    # TODO: reshpae added to nn_tools layer
    dims = list(args)
    dims[0] = 0  # the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x


def _mean(input, *args, **kwargs):
    x = raw_mean(input, *args, **kwargs)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mean')
    top_blobs = log.add_blobs([x], name='mean_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reduction',
                                  bottom=[log.blobs(input)], top=top_blobs)
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    layer.param.reduction_param.operation = 4
    layer.param.reduction_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _add(input, *args):
    # check if add a const value
    if isinstance(args[0], int):
        print('value: ',args[0])
        x = raw__add__(input, *args)
        #x = raw(input)
        layer_name = log.add_layer(name='scale')
        log.add_blobs([x], name='Scale_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                       bottom=[log.blobs(input)], top=[log.blobs(x)])
        dim = x.shape[1]
        layer.param.scale_param.bias_term = True
        weight = np.ones(dim, dtype=np.float32)
        bias = args[0] * np.ones(dim, dtype=np.float32)
        layer.add_data(weight, bias)
        log.cnet.add_layer(layer)
        return x
    # otherwise add a tensor
    x = raw__add__(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def _iadd(input, *args):
    x = raw__iadd__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def _sub(input, *args):
    x = raw__sub__(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


def _isub(input, *args):
    x = raw__isub__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def _mul(input, *args):
    x = raw__sub__(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    log.cnet.add_layer(layer)
    return x


def _imul(input, *args):
    x = raw__isub__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


def _adaptive_avg_pool2d(raw, input, output_size):
    _output_size = _list_with_default(output_size, input.size())
    x = raw(input, _output_size)
    _pool('ave', raw, input, x, input.shape[2], input.shape[2], 0, False)
    return x


# 核心组件，通过该类，实现对torch的function中的operators的输入，输出以及参数的读取
class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace function
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if not NET_INITTED:
            return self.raw(*args, **kwargs)
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in layer_names:
                    log.pytorch_layer_name = layer_names[layer]
                    print(layer_names[layer])
                    break
        out = self.obj(self.raw, *args, **kwargs)
        # if isinstance(out,Variable):
        #     out=[out]
        return out


F.conv2d = Rp(F.conv2d, _conv2d)
F.linear = Rp(F.linear, _linear)
F.relu = Rp(F.relu, _relu)

F.leaky_relu = Rp(F.leaky_relu, _leaky_relu)
F.max_pool2d = Rp(F.max_pool2d, _max_pool2d)
F.avg_pool2d = Rp(F.avg_pool2d, _avg_pool2d)
F.dropout = Rp(F.dropout, _dropout)
F.threshold = Rp(F.threshold, _threshold)
F.prelu = Rp(F.prelu, _prelu)
F.batch_norm = Rp(F.batch_norm, _batch_norm)
F.instance_norm = Rp(F.instance_norm, _instance_norm)
F.softmax = Rp(F.softmax, _softmax)
F.conv_transpose2d = Rp(F.conv_transpose2d, _conv_transpose2d)
F.interpolate = Rp(F.interpolate, _interpolate)
F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)

torch.split = Rp(torch.split, _split)
torch.max = Rp(torch.max, _max)
torch.cat = Rp(torch.cat, _cat)
torch.sigmoid = Rp(torch.sigmoid, _sigmoid)

# TODO: other types of the view function
try:
    raw_view = Variable.view
    Variable.view = _view
    raw_mean = Variable.mean
    Variable.mean = _mean
    raw__add__ = Variable.__add__
    Variable.__add__ = _add
    raw__iadd__ = Variable.__iadd__
    Variable.__iadd__ = _iadd
    raw__sub__ = Variable.__sub__
    Variable.__sub__ = _sub
    raw__isub__ = Variable.__isub__
    Variable.__isub__ = _isub
    raw__mul__ = Variable.__mul__
    Variable.__mul__ = _mul
    raw__imul__ = Variable.__imul__
    Variable.__imul__ = _imul
except:
    # for new version 0.4.0 and later version
    for t in [torch.Tensor]:
        raw_view = t.view
        t.view = _view
        raw_mean = t.mean
        t.mean = _mean
        raw__add__ = t.__add__
        t.__add__ = _add
        raw__iadd__ = t.__iadd__
        t.__iadd__ = _iadd
        raw__sub__ = t.__sub__
        t.__sub__ = _sub
        raw__isub__ = t.__isub__
        t.__isub__ = _isub
        raw__mul__ = t.__mul__
        t.__mul__ = _mul
        raw__imul__ = t.__imul__
        t.__imul__ = _imul


def trans_net(net, input_var, name='TransferedPytorchModel'):
    print('Starting Transform, This will take a while')
    log.init([input_var])
    log.cnet.net.name = name
    log.cnet.net.input.extend([log.blobs(input_var)])
    log.cnet.net.input_dim.extend(input_var.size())
    global NET_INITTED
    NET_INITTED = True
    for name, layer in net.named_modules():
        layer_names[layer] = name
    print("torch ops name:", layer_names)
    out = net.forward(input_var)
    print('Transform Completed')


def save_prototxt(save_name):
    log.cnet.save_prototxt(save_name)


def save_caffemodel(save_name):
    log.cnet.save(save_name)
