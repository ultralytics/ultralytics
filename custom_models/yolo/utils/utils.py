import os
import torch
import cv2
import numpy as np

# Tensor.element_size() → int
# Returns the size in bytes of an individual element.
import config


def check_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def strip_model(model):
    model.half()
    for p in model.parameters():
        p.requires_grid = False

# Tensor.element_size() → int
# Returns the size in bytes of an individual element.
def save_model(model, folder_path, file_name):
    ckpt = {}
    ckpt["model"] = model
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("Saving Model...")
    torch.save(ckpt, os.path.join(folder_path, file_name))

def export_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      "netron_onnx_files/yolov5m_mine.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      opset_version=11
                      )


def save_checkpoint(state, folder_path, filename, epoch):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        os.makedirs(path)

    print("=> Saving checkpoint...")
    torch.save(state, os.path.join(path, f"checkpoint_epoch_{str(epoch)}.pth.tar"))


def load_model_checkpoint(model_name, model, last_epoch):
    folder = os.listdir(os.path.join("SAVED_CHECKPOINT", model_name))
    
    ckpt_name = f"checkpoint_epoch_{last_epoch}.pth.tar"
    print(f"==> loading model weights stored in {ckpt_name} ")
    
    checkpoint = torch.load(os.path.join("SAVED_CHECKPOINT", model_name, ckpt_name), map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


def load_optim_checkpoint(model_name, optim, last_epoch):
    folder = os.listdir(os.path.join("SAVED_CHECKPOINT", model_name))

    ckpt_name = f"checkpoint_epoch_{last_epoch}.pth.tar"
    print(f"==> loading optimizer stored in {ckpt_name}")
    
    checkpoint = torch.load(os.path.join("SAVED_CHECKPOINT", model_name, ckpt_name), map_location=config.DEVICE)
    optim.load_state_dict(checkpoint["optimizer"])


def resize_image(image, output_size):
    # output size is [width, height]
    return cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)

def coco91_2_coco80(label):
    # idx & labels below are not present in MS_COCO
    """11: 'street sign', 25: 'hat', 28: 'shoe', 29: 'eye glasses', 44: 'plate', 65: 'mirror',
    67: 'window', 68: 'desk', 70: 'door', 82: 'blender', 90: 'hairbrush'"""
    if 11 < label <= 25:
        return label - 1
    elif 25 < label <= 28:
        return label - 2
    elif 28 < label <= 29:
        return label - 3
    elif 29 < label <= 44:
        return label - 4
    elif 44 < label <= 65:
        return label - 5
    elif 65 < label <= 67:
        return label - 6
    elif 67 < label <= 68:
        return label - 7
    elif 68 < label <= 70:
        return label - 8
    elif 70 < label <= 82:
        return label - 9
    elif 82 < label <= 90:
        return label - 10
    elif label > 90:
        return label - 11
    else:
        return label

# Ultralytics
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


