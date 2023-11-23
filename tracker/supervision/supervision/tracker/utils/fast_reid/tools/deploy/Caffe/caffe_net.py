from __future__ import absolute_import
from . import caffe_pb2 as pb
import google.protobuf.text_format as text_format
import numpy as np
from .layer_param import Layer_param

class _Net(object):
    def __init__(self):
        self.net=pb.NetParameter()

    def layer_index(self,layer_name):
        # find a layer's index by name. if the layer was found, return the layer position in the net, else return -1.
        for i, layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                return i

    def add_layer(self,layer_params,before='',after=''):
        # find the before of after layer's position
        index = -1
        if after != '':
            index = self.layer_index(after) + 1
        if before != '':
            index = self.layer_index(before)
        new_layer = pb.LayerParameter()
        new_layer.CopyFrom(layer_params.param)
        #insert the layer into the layer protolist
        if index != -1:
            self.net.layer.add()
            for i in range(len(self.net.layer) - 1, index, -1):
                self.net.layer[i].CopyFrom(self.net.layer[i - 1])
            self.net.layer[index].CopyFrom(new_layer)
        else:
            self.net.layer.extend([new_layer])

    def remove_layer_by_name(self,layer_name):
        for i,layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                del self.net.layer[i]
                return
        raise(AttributeError, "cannot found layer %s" % str(layer_name))

    def get_layer_by_name(self, layer_name):
        # get the layer by layer_name
        for layer in self.net.layer:
            if layer.name == layer_name:
                return layer
        raise(AttributeError, "cannot found layer %s" % str(layer_name))

    def save_prototxt(self,path):
        prototxt=pb.NetParameter()
        prototxt.CopyFrom(self.net)
        for layer in prototxt.layer:
            del layer.blobs[:]
        with open(path,'w') as f:
            f.write(text_format.MessageToString(prototxt))

    def layer(self,layer_name):
        return self.get_layer_by_name(layer_name)

    def layers(self):
        return list(self.net.layer)



class Prototxt(_Net):
    def __init__(self,file_name=''):
        super(Prototxt,self).__init__()
        self.file_name=file_name
        if file_name!='':
            f = open(file_name,'r')
            text_format.Parse(f.read(), self.net)
            pass

    def init_caffemodel(self,caffe_cmd_path='caffe'):
        """
        :param caffe_cmd_path: The shell command of caffe, normally at <path-to-caffe>/build/tools/caffe
        """
        s=pb.SolverParameter()
        s.train_net=self.file_name
        s.max_iter=0
        s.base_lr=1
        s.solver_mode = pb.SolverParameter.CPU
        s.snapshot_prefix='./nn'
        with open('/tmp/nn_tools_solver.prototxt','w') as f:
            f.write(str(s))
        import os
        os.system('%s train --solver /tmp/nn_tools_solver.prototxt'%caffe_cmd_path)

class Caffemodel(_Net):
    def __init__(self, file_name=''):
        super(Caffemodel,self).__init__()
        # caffe_model dir
        if file_name!='':
            f = open(file_name,'rb')
            self.net.ParseFromString(f.read())
            f.close()

    def save(self, path):
        with open(path,'wb') as f:
            f.write(self.net.SerializeToString())

    def add_layer_with_data(self,layer_params,datas, before='', after=''):
        """
        Args:
            layer_params:A Layer_Param object
            datas:a fixed dimension numpy object list
            after: put the layer after a specified layer
            before: put the layer before a specified layer
        """
        self.add_layer(layer_params,before,after)
        new_layer =self.layer(layer_params.name)

        #process blobs
        del new_layer.blobs[:]
        for data in datas:
            new_blob=new_layer.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))

    def get_layer_data(self,layer_name):
        layer=self.layer(layer_name)
        datas=[]
        for blob in layer.blobs:
            shape=list(blob.shape.dim)
            data=np.array(blob.data).reshape(shape)
            datas.append(data)
        return datas

    def set_layer_data(self,layer_name,datas):
        # datas is normally a list of [weights,bias]
        layer=self.layer(layer_name)
        for blob,data in zip(layer.blobs,datas):
            blob.data[:]=data.flatten()
            pass

class Net():
    def __init__(self,*args,**kwargs):
        raise(TypeError,'the class Net is no longer used, please use Caffemodel or Prototxt instead')