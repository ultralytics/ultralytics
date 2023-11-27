import lmdb
from Caffe import caffe_pb2 as pb2
import numpy as np

class Read_Caffe_LMDB():
    def __init__(self,path,dtype=np.uint8):

        self.env=lmdb.open(path, readonly=True)
        self.dtype=dtype
        self.txn=self.env.begin()
        self.cursor=self.txn.cursor()

    @staticmethod
    def to_numpy(value,dtype=np.uint8):
        datum = pb2.Datum()
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=dtype)
        data = flat_x.reshape(datum.channels, datum.height, datum.width)
        label=flat_x = datum.label
        return data,label

    def iterator(self):
        while True:
            key,value=self.cursor.key(),self.cursor.value()
            yield self.to_numpy(value,self.dtype)
            if not self.cursor.next():
                return

    def __iter__(self):
        self.cursor.first()
        it = self.iterator()
        return it

    def __len__(self):
        return int(self.env.stat()['entries'])
