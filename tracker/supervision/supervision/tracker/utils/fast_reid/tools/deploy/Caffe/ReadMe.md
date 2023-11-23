# The Caffe in nn_tools Provides some convenient API
If there are some problem in parse your prototxt or caffemodel, Please replace
the caffe.proto with your own version and compile it with command
                   `protoc --python_out ./ caffe.proto`

## caffe_net.py
Using `from nn_tools.Caffe import caffe_net` to import this model
### Prototxt
+ `net=caffe_net.Prototxt(file_name)` to open a prototxt file
+ `net.init_caffemodel(caffe_cmd_path='caffe')` to generate a caffemodel file in the current work directory \
if your `caffe` cmd not in the $PATH, specify your caffe cmd path by the `caffe_cmd_path` kwargs.
### Caffemodel
+ `net=caffe_net.Caffemodel(file_name)` to open a caffemodel
+ `net.save_prototxt(path)` to save the caffemodel to a prototxt file (not containing the weight data)
+ `net.get_layer_data(layer_name)` return the numpy ndarray data of the layer
+ `net.set_layer_date(layer_name, datas)` specify the data of one layer in the caffemodel .`datas` is normally a list of numpy ndarray `[weights,bias]`
+ `net.save(path)` save the changed caffemodel
### Functions for both Prototxt and Caffemodel
+ `net.add_layer(layer_params,before='',after='')` add a new layer with `Layer_Param` object
+ `net.remove_layer_by_name(layer_name)` 
+ `net.get_layer_by_name(layer_name)` or `net.layer(layer_name)` get the raw Layer object defined in caffe_pb2
