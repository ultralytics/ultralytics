import io
import os.path
import time

import requests
import torch

from ultralytics import YOLO
import cv2
import numpy as np
import math
from equilib import equi2pers
from PIL import Image, ImageDraw, ImageFont
from mmseg.apis import MMSegInferencer


class PersImage:
    def __init__(self, pitch, yaw, pers_img):
        self.pitch = pitch
        self.yaw = yaw
        self.pers_img = pers_img


class Rectangle:
    def __init__(self, p1, p2, cls, conf, pitch, yaw):
        self.p1 = p1
        self.p2 = p2
        self.cls = cls
        self.conf = conf
        self.pitch = pitch
        self.yaw = yaw

    def intersect_with(self, other):
        r1_lft_btm_x = self.p1[0]
        r1_lft_btm_y = self.p2[1]
        r1_rt_top_x = self.p2[0]
        r1_rt_top_y = self.p1[1]

        r2_lft_btm_x = other.p1[0]
        r2_lft_btm_y = other.p2[1]
        r2_rt_top_x = other.p2[0]
        r2_rt_top_y = other.p1[1]

        cx1 = max(r1_lft_btm_x, r2_lft_btm_x)
        cy1 = max(r1_lft_btm_y, r2_lft_btm_y)
        cx2 = min(r1_rt_top_x, r2_rt_top_x)
        cy2 = min(r1_rt_top_y, r2_rt_top_y)

        # cy1 >= cy2的判断是因为坐标系是left top，y轴向下增长
        return self.cls == other.cls and cx1 <= cx2 and cy1 >= cy2

    def union(self, other):
        self.p1[0] = min(self.p1[0], other.p1[0])
        self.p1[1] = min(self.p1[1], other.p1[1])
        self.p2[0] = max(self.p2[0], other.p2[0])
        self.p2[1] = max(self.p2[1], other.p2[1])
        self.conf = max(self.conf, other.conf)


def screen_to_equirectangular(x, y, screen_width, screen_height, fov, yaw, pitch, equi_width, equi_height):
    # 将屏幕坐标(x, y)转换到NDC坐标(-1 to 1)
    nx = (x / screen_width) * 2 - 1
    ny = (y / screen_height) * 2 - 1

    # FOV的一半的切线值，用于计算z坐标
    t = np.tan(np.radians(fov / 2))

    # 逆向计算出对应的方向向量
    direction = np.array([t * nx, t * ny, 1])
    # 转为单位向量
    direction = direction / np.linalg.norm(direction)

    # 建立旋转矩阵，考虑偏航角和俯仰角
    cos_yaw, sin_yaw = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cos_pitch, sin_pitch = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

    rotation_matrix = np.array([
        [cos_yaw, 0, -sin_yaw],
        [0, 1, 0],
        [sin_yaw, 0, cos_yaw]
    ]) @ np.array([
        [1, 0, 0],
        [0, cos_pitch, sin_pitch],
        [0, -sin_pitch, cos_pitch]
    ])

    # 将方向向量旋转到最终的方向
    final_dir = rotation_matrix @ direction

    # 计算球面坐标
    longitude = np.arctan2(final_dir[0], final_dir[2])
    latitude = np.arcsin(final_dir[1])

    # 转换为等距平面坐标
    ex = (longitude + np.pi) / (2 * np.pi) * equi_width
    ey = (latitude + np.pi / 2) / np.pi * equi_height

    return [int(ex), int(ey)]

# Load a model
model = YOLO('runs/detect/train33/weights/best.pt')  # pretrained YOLOv8n model
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')

def predict_result(image):
    ndarr = image
    classes = inferencer.visualizer.dataset_meta['classes']
    num_classes = len(classes)
    palette = inferencer.visualizer.dataset_meta['palette']
    ids = np.unique(ndarr)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]
    shape = np.append(np.array(ndarr.shape), 3)
    result = np.empty(shape, np.uint8)
    for i in range(ndarr.shape[0]):
        for j in range(ndarr.shape[1]):
            pred_category = ndarr[i, j]
            result[i, j] = palette[pred_category]
    mask = Image.fromarray(result)
    return mask
# Run batched inference on a list of images
images = [
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/52f3a0273dfb466e85e6bf59ade05c2e.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6176dca744e64bd6bf8a02dd92f466eb.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/057c166dcc8648e69cba69009a8535b4.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0b3f8b79969940528ab1cf6a43ca83a9.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9cf26a02681f44f68b718762cd0f5494.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/aa00a97db0a34c0e90eea6d393cedde2.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7868ef382f84429fb79047fba3978b67.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/723fbb77ae71495e8407265cb3d8f7db.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/33fa8f17405f415195055f6a714f4c09.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/809d5a96fee24fc294fb4a8d5445412a.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d5c8a742f7464f96979633f0dc2f10aa.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e1e599f026834cd0ae7f0a714123175d.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d92593ffbcf74d80aa12afaa43a26584.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/74483c83efb84cf7a41f55c314bffcb8.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/61523fb326234be194e5d41afa06b17b.jpg'
]

img_names = [x.split('/')[-1].split('.')[0] for x in images]

name_dict = {
    0: '路面破损',
    # 1: '沿街晾晒',
    # 2: '垃圾满冒',
    # 3: '乱扔垃圾',
    # 4: '垃圾正常盛放',
}

confs = [0.1]


def cvt_coord(yaw, pitch):
    _u = 0.5 + 0.5 * (-yaw / math.pi)
    _v = 0.5 + (pitch / math.pi)
    return _u, _v


for idx_conf in confs:
    if not os.path.exists(f'conf_{idx_conf}'):
        os.mkdir(f'conf_{idx_conf}')
    for img_idx, image_url in enumerate(images):
        begin = time.time()
        now = time.time()
        res = requests.get(image_url)
        img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.imread(image_url)
        if img is None:
            continue
        # img = cv2.imread(image_url)
        # cv2.imwrite(f'det/{img_names[img_idx]}_orig_result.jpg', img)
        print(f'下载图片耗时{time.time() - now:.2f}s')
        # img = e2c(img, face_w=int(img.shape[0] / 3))
        now = time.time()
        (height, width, depth) = img.shape
        equi_img = np.transpose(img, (2, 0, 1))
        equi_img = torch.tensor(equi_img)
        pers_should_height = int(height / 4)
        pers_should_width = int(width / 4)
        pers_height = 480
        pers_width = 640
        pitch = math.radians(0)
        fov_deg = 90.0
        rectangles = []
        img_PIL = Image.fromarray(img[..., ::-1])  # 转成 PIL 格式
        draw = ImageDraw.Draw(img_PIL)  # 创建绘制对象
        # 俯仰角从0到30度，每次转动5度
        for j in range(1):
            now = time.time()
            yaw = math.pi
            pers_imgs = []
            # 偏航角每次转
            # 动10度
            dir = f'pitch_{math.ceil(math.degrees(pitch))}/{img_names[img_idx]}'
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(os.path.join(dir, '标注图')):
                os.mkdir(os.path.join(dir, '标注图'))
            if not os.path.exists(os.path.join(dir, '原图')):
                os.mkdir(os.path.join(dir, '原图'))
            for i in range(36):
                rots = {
                    'roll': 0.,
                    'pitch': pitch,  # rotate vertical
                    'yaw': yaw,  # rotate horizontal
                }
                # Run equi2pers
                fov_deg = 90.0
                pers_height = 320
                pers_width = 320
                # pers_height = pers_should_height
                # pers_width = pers_should_width
                pers_img = equi2pers(
                    equi=equi_img,
                    rots=rots,
                    height=pers_height,
                    width=pers_width,
                    fov_x=fov_deg,
                    mode="bilinear",
                )
                cube_result = np.ascontiguousarray(np.transpose(pers_img, (1, 2, 0)))
                cv2.imshow('img', cube_result)
                cv2.waitKey(0)
                pers_imgs.append(PersImage(pitch, yaw, cube_result))
                _u, _v = cvt_coord(yaw, pitch)
                center = np.array([int(pers_width / 2), int(pers_height / 2)])
                vec = np.array([10 - center[0], 10 - center[1]])
                vec = [int(vec[0] * pers_should_width / pers_width), int(vec[1] * pers_should_height / pers_height)]
                # 得到当前视角中心点在等矩平面上的坐标
                center_equi_pos = np.array([int(_u * width), int(_v * height)])
                point = center_equi_pos + vec
                point[0] = point[0] + width if point[0] < 0 else point[0]
                point = [a for a in point]
                yaw -= np.pi / 18
                # cv2.imshow('img', cube_result)
                # cv2.waitKey(0)
            pitch += math.radians(5)
            print(f'转动视角耗时{time.time() - now}秒')

            results = model.predict([elem.pers_img for elem in pers_imgs], conf=0.3, imgsz=640)
            for idx, result in enumerate(results):
                pers_image = pers_imgs[idx]
                orig_image = Image.fromarray(pers_image.pers_img[..., ::-1])  # 转成 PIL 格式
                orig_draw = ImageDraw.Draw(orig_image)
                # cv2.imwrite(f'{dir}/原图/origin_img_{idx}.jpg', pers_image.pers_img)
                for cls, box, conf in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
                    cls_np = int(cls.cpu().detach().numpy().item())
                    if cls_np not in name_dict.keys():
                        continue
                    box_np = box.cpu().detach().numpy().squeeze()
                    conf_np = conf.cpu().detach().numpy().item()
                    p1 = (box_np[0], box_np[1])
                    p2 = (box_np[2], box_np[3])
                    # if (p2[0] - p1[0]) * (p2[1] - p1[1]) < 100:
                    #     continue
                    left_top = screen_to_equirectangular(box_np[0], box_np[1], pers_width, pers_height, fov_deg,
                                                         math.degrees(pers_image.yaw), math.degrees(pers_image.pitch), width, height)
                    right_bottom = screen_to_equirectangular(box_np[2], box_np[3], pers_width, pers_height, fov_deg,
                                                             math.degrees(pers_image.yaw), math.degrees(pers_image.pitch), width, height)
                    if left_top[0] > right_bottom[0]:
                        # 如果x1大于x2且它俩之间的距离相差大于一屏，则说明标注框跨过了接缝，需要将x1进行偏移
                        if left_top[0] - right_bottom[0] > pers_should_width:
                            left_top[0] -= width
                        else:
                            left_top[0], right_bottom[0] = right_bottom[0], left_top[0]
                    if left_top[1] > right_bottom[1]:
                        left_top[1], right_bottom[1] = right_bottom[1], left_top[1]
                    # cv2.imshow('img', cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(0)

                    orig_draw.rectangle(xy=(p1, p2), fill=None, outline='red', width=5)
                    # cv2.rectangle(img=img, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=10)
                    font = ImageFont.truetype(font='wqy-zenhei.ttc', size=40)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
                    orig_draw.rectangle(((p1[0], p1[1] - 50), (p1[0] + 300, p1[1])),
                                        fill=(255, 0, 0), )
                    name = f'{name_dict[cls_np]} {conf_np:.2f}'
                    orig_draw.text(xy=(p1[0], p1[1] - font.size - 10), text=name, font=font,
                                   fill=(255, 255, 255))
                    cv2.imshow('img', cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)
                    r = Rectangle(left_top, right_bottom, cls_np, conf_np, pers_image.pitch, pers_image.yaw)
                    r.img_idx = idx
                    r.pers_p1 = [box_np[0], box_np[1]]
                    r.pers_p2 = [box_np[2], box_np[3]]
                    rectangles.append(r)
                # print(f'保存标注图第{idx}张')
                # cv2.imwrite(f'{dir}/标注图/labeled_img_{idx}.jpg', cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR))
        # merged = []
        # for rect1 in rectangles:
        #     # 如果该矩形已被其他矩形合并过则不需要再处理
        #     if rect1 in merged:
        #         continue
        #     for rect2 in rectangles:
        #         if rect2 in merged or rect1 == rect2:
        #             continue
        #         # 如果两个矩形相交则合并，被合并的矩形标记为已被合并
        #         if rect1.intersect_with(rect2):
        #             rect1.union(rect2)
        #             merged.append(rect2)
        # rectangles = list(filter(lambda rect: rect not in merged, rectangles))
        for rectangle in rectangles:
            pitch = rectangle.pitch
            yaw = rectangle.yaw
            draw.rectangle(xy=((rectangle.p1[0], rectangle.p1[1]), (rectangle.p2[0], rectangle.p2[1])), fill=None,
                           outline='red', width=10)
            # cv2.rectangle(img=img, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=10)
            font = ImageFont.truetype(font='wqy-zenhei.ttc', size=40)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
            draw.rectangle(((rectangle.p1[0], rectangle.p1[1] - 50), (rectangle.p1[0] + 300, rectangle.p1[1])),
                           fill=(255, 0, 0), )
            # print(left_top, right_bottom)
            name = f'{name_dict[rectangle.cls]} {rectangle.conf:.2f}'
            draw.text(xy=(rectangle.p1[0], rectangle.p1[1] - font.size - 10), text=name, font=font,
                      fill=(255, 255, 255))
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 再转成 OpenCV 的格式，记住 OpenCV 中通道排布是 BGR
        # cv2.imwrite(f'det_merged/{img_names[img_idx]}_det_result.jpg', img)
        cv2.imwrite(f'det_result.jpg', img)

        # seg_img = requests.get(image_url + '?x-oss-process=image/resize,h_1024,m_lfit')
        # seg_img_arr = np.array(Image.open(io.BytesIO(seg_img.content)))
        # mask = predict_result(inferencer(seg_img_arr, show=False)['predictions'])
        # mask.save(f'seg/{img_names[img_idx]}_seg_result.jpg')


        print(f'总耗时:{time.time() - begin}秒')