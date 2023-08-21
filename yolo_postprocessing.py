import numpy as np


class YoloV6:
    def generate_anchor_grid(inp_shape=(640, 640), strides=(8, 16, 32)):
        # get num passes for each stride -- length / stride
        passes = np.int32(np.divide(inp_shape, np.tile(strides, (2, 1)).T))

        a = np.zeros((np.sum(np.prod(passes, axis=1)), 2))

        start = 0
        end = 0

        for row in passes:
            y, x = tuple(row)
            end += x * y

            a[start:end, 0] = np.tile(np.arange(x), (1, y)) + 0.5
            a[start:end, 1] = np.tile(np.arange(y).reshape((-1, 1)), (1, x)).reshape((-1)) + 0.5

            start += x * y

        return a 

    def generate_xy_mul_grid(inp_shape=(640, 640), strides=(8, 16, 32)):
        inp_shape = np.array(inp_shape)
        xy_strides = np.tile(strides, (2, 1)).T
        passes = np.int32(np.divide(inp_shape, xy_strides)) # pass = side length / stride length

        return np.hstack([np.ones(np.prod(p)) * stride for p, stride in zip(passes, strides)]).reshape((-1, 1))

    def process(boxes: np.array, class_probs: np.array, anchor_points: np.array, xy_mul_grid: np.array):
        assert boxes.shape == (1, 4, class_probs.shape[-1])
        assert anchor_points.shape == (boxes.shape[-1], 2)
        assert xy_mul_grid.shape == (boxes.shape[-1], 1)

        boxes = boxes.transpose((0, 2, 1))
        a = boxes[:,:,:2]
        b = boxes[:,:,2:]

        xy = (anchor_points + (b - a) / 2) * xy_mul_grid
        wh = (a + b) * xy_mul_grid

        xywh = np.concatenate((xy, wh), axis=-1)

        class_probs = class_probs.transpose((0, 2, 1))

        return np.concatenate((xywh, np.ones((1, xywh.shape[1], 1)), class_probs), axis=-1)


class YoloV8:
    def generate_anchor_grid(inp_shape=(640, 640), strides=(8, 16, 32)):
        a = YoloV6.generate_anchor_grid(inp_shape, strides)
        return np.expand_dims(a.T, axis=0)


    def generate_xy_mul_grid(inp_shape=(640, 640), strides=(8, 16, 32)):
        return YoloV6.generate_xy_mul_grid(inp_shape, strides).T

    def process(tensor: np.array, xy_anchor_grid: np.array, xy_mul_grid: np.array, anchors_per_stride: int = 16, num_classes: int = 80):
        assert tensor.shape[1] == 4 * anchors_per_stride + num_classes
        assert xy_anchor_grid.shape == (1, 2, tensor.shape[-1])
        assert xy_mul_grid.shape == (1, tensor.shape[-1])

        boxes = tensor[:, :64, :]
        class_probs = tensor[:, 64:, :]

        boxes = boxes.reshape((1, 4, anchors_per_stride, -1))  # 4 is num coords per box

        boxes = np.exp(boxes) / np.sum(np.exp(boxes), axis=2, keepdims=True)  # softmax
        boxes = boxes.transpose((0, 2, 1, 3))

        boxes = np.sum(boxes * np.arange(anchors_per_stride).reshape((1, anchors_per_stride, 1, 1)), axis=1)  # conv

        slice_a = boxes[:, :2, :]
        slice_b = boxes[:, 2:4, :]

        # get xy and wh
        xy = (xy_anchor_grid + (slice_b - slice_a) / 2) * xy_mul_grid
        wh = (slice_b + slice_a) * xy_mul_grid

        xywh = np.concatenate((xy, wh), axis=1)

        class_probs = 1.0 / (1.0 + np.exp(-class_probs))  # sigmoid on class probs

        out = np.concatenate((xywh, class_probs), axis=1)

        return out