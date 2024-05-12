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
# from mmseg.apis import MMSegInferencer


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
    t = 1

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
model = YOLO('runs/detect/train55/weights/best.pt')  # pretrained YOLOv8n model
# inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')


# def predict_result(image):
#     ndarr = image
#     classes = inferencer.visualizer.dataset_meta['classes']
#     num_classes = len(classes)
#     palette = inferencer.visualizer.dataset_meta['palette']
#     ids = np.unique(ndarr)[::-1]
#     legal_indices = ids < num_classes
#     ids = ids[legal_indices]
#     labels = np.array(ids, dtype=np.int64)
#
#     colors = [palette[label] for label in labels]
#     shape = np.append(np.array(ndarr.shape), 3)
#     result = np.empty(shape, np.uint8)
#     for i in range(ndarr.shape[0]):
#         for j in range(ndarr.shape[1]):
#             pred_category = ndarr[i, j]
#             result[i, j] = palette[pred_category]
#     mask = Image.fromarray(result)
#     return mask


# Run batched inference on a list of images
images = [
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/52f3a0273dfb466e85e6bf59ade05c2e.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6176dca744e64bd6bf8a02dd92f466eb.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/057c166dcc8648e69cba69009a8535b4.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0b3f8b79969940528ab1cf6a43ca83a9.jpg',
    # 'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9cf26a02681f44f68b718762cd0f5494.jpg',
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
    # 'img_662504a04dbc57656cfce1ed/926c8249e8454d2c83ba81f4b06b30c0.jpg',
    # 'img_662504a04dbc57656cfce1ed/4ae3658f9f794ca08b462c6fe6cc7390.jpg',
    # 'img_662504a04dbc57656cfce1ed/e3a4dd496cc142e5977fa841c1cdb588.jpg',
    # 'img_662504a04dbc57656cfce1ed/748f05324f1444fd8f6b232300ccec6e.jpg',
    # 'img_662504a04dbc57656cfce1ed/51f9d80eaa8746919ab777ba974e8678.jpg',
    # 'img_662504a04dbc57656cfce1ed/cb3cdca3ddbe4805af66a4b88bb057d9.jpg',
    # 'img_662504a04dbc57656cfce1ed/fe146e68a6c84c868243a01bec88e4c8.jpg',
    # 'img_662504a04dbc57656cfce1ed/59117f01c8eb4579afbb04b6e9163925.jpg',
    # 'img_662504a04dbc57656cfce1ed/013e92db2aa0445b9b7c606d53c2c923.jpg',
    # 'img_662504a04dbc57656cfce1ed/6ee72c73a2f44f689349f1e041ea233d.jpg',
    # 'img_662b005089d953346ed1c475/33b749b9bd2f40a4840e17d5254e16ed.jpg',
    # 'img_662b005089d953346ed1c475/ce965bad47734fcbb026788bba50e1f7.jpg',
    # 'img_662b005089d953346ed1c475/a6a88f1785fe4042b24338b565c1fd63.jpg',
    # 'img_662b005089d953346ed1c475/f274bc26311c4ab28b2444082f2a8d49.jpg',
    # 'img_662b005089d953346ed1c475/72fadd4ee38a4ad392332aca03c85faa.jpg',
    # 'img_662b005089d953346ed1c475/06d54efd006e490d8aa5603260ccb4c2.jpg',
    # 'img_662b005089d953346ed1c475/f85c77a28cba414dbd1ed9a5a24307d9.jpg',
    # 'img_662b005089d953346ed1c475/0e737a627235479c8ff3a1a76100bb9f.jpg',
    # 'img_662b005089d953346ed1c475/0c2d58d8f45d4a4184d9df65d77a61f5.jpg',
    # 'img_662b005089d953346ed1c475/85a20c16d6ab41d18661271a44f88aed.jpg',
    # 'img_662680943071dd4f07eca23c/40664431b6574654bb77b9ebebfbbd55.jpg',
    # 'img_662680943071dd4f07eca23c/9a686709e7584f9d8ca10578590416c7.jpg',
    # 'img_662680943071dd4f07eca23c/2afed2504eaa4f1290daafe0953591b0.jpg',
    # 'img_662680943071dd4f07eca23c/395d151408804ae981d5d3a636d12fdd.jpg',
    # 'img_662680943071dd4f07eca23c/3980c02e40054692ae46c3ad6d55bbea.jpg',
    # 'img_662680943071dd4f07eca23c/be5aba00f33f4814ad1f6cfd8f2b0ba2.jpg',
    # 'img_662680943071dd4f07eca23c/3f08a283daf74389b21743f91afdeb38.jpg',
    # 'img_662680943071dd4f07eca23c/0761f55764f248e586bca24dcfca41dd.jpg',
    # 'img_662680943071dd4f07eca23c/3071d1779de24408a740633e8aa16f64.jpg',
    # 'img_662680943071dd4f07eca23c/be1c80cb99414663bdaa7500b74c8456.jpg',
    # 'img_663c600f9036c94341364414/dcb8e696cc0f4218a43ce5bb94aaebd0.jpg',
    # 'img_663c600f9036c94341364414/59c57873ff1c46449017f7e58d0f0e1f.jpg',
    # 'img_663c600f9036c94341364414/ba39f9338c6f43298a03d923833ec62b.jpg',
    # 'img_663c600f9036c94341364414/8f855773ff664bf99d9a5d990c09cd08.jpg',
    # 'img_663c600f9036c94341364414/3211f225c2d44afe85f301f55b0c09d9.jpg',
    # 'img_663c600f9036c94341364414/0cec5eb87ce14537b666b09dcb3c8849.jpg',
    # 'img_663c600f9036c94341364414/4f13bf389b5d49d8b698b630be4b6e25.jpg',
    # 'img_663c600f9036c94341364414/d7fea3a81c0d4783b9f3dd3fb5c1ffee.jpg',
    # 'img_663c600f9036c94341364414/3d9ed423e76e4074b47fcfafd3af5bb3.jpg',
    # 'img_663c600f9036c94341364414/5b619abf3b29451f9d7850789f3577a2.jpg',
    # 'img_662680703071dd4f07eca23b/226134332a1943f2b7ddd8f93e94c439.jpg',
    # 'img_662680703071dd4f07eca23b/5dd5f033eb134b1ab3c40a082c28bc61.jpg',
    # 'img_662680703071dd4f07eca23b/77f7cea57d564527832efb59017481f4.jpg',
    # 'img_662680703071dd4f07eca23b/3ccce744cc6e4bb8a90d60f54c3525de.jpg',
    # 'img_662680703071dd4f07eca23b/1f1c63f7dcbe43f7ba174688d2bfc01f.jpg',
    # 'img_662680703071dd4f07eca23b/1f76e22065e9433fa63d9b3843e09f17.jpg',
    # 'img_662680703071dd4f07eca23b/ec2869c2311b4129a830489f91440249.jpg',
    # 'img_662680703071dd4f07eca23b/08e00b6412c847ecb18c84b81449ec15.jpg',
    # 'img_662680703071dd4f07eca23b/efa8564101ec4906843e8519c6d5f44f.jpg',
    # 'img_662680703071dd4f07eca23b/26d62670f5c84f1e92c0402b0e824130.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/1ff0ca6dd1354fe6ad651a3fb929356f.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/9f2a417c4e1b488a9f340c33f38f7eeb.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/2d8831de52814dc18909d788e05d438a.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/1cc1beb7ae3f46a29b7a56cc1478d41e.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/12ca6d7291734bc2924d64f9097e4bef.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/afb4cb43e14941e7b074123c040f747b.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/8cac48fc625543839aa1bae0898d88a4.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/5283352cbdf343bd8e4ce197cbbd084a.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/7289a96720c34e3a83adf62867fc6615.jpg',
    # 'img_6639c3e6e1fb1f61863c2359/a0302eaf6dc84a84842970feb9cab50f.jpg',
    # 'img_66268a793071dd4f07eca241/71e86432ce6d480bafa2c43493ec4568.jpg',
    # 'img_66268a793071dd4f07eca241/789a03d9970d433cbdfb62e24498454d.jpg',
    # 'img_66268a793071dd4f07eca241/9e6b4cce1bf84c40829a4cad796fe967.jpg',
    # 'img_66268a793071dd4f07eca241/2107cc43c470409a9fcaa120e100476e.jpg',
    # 'img_66268a793071dd4f07eca241/6e0aeebd172640b88c72d3fc88f3fab9.jpg',
    # 'img_66268a793071dd4f07eca241/eab9eed230cc484da5ea07a3d7cf2bef.jpg',
    # 'img_66268a793071dd4f07eca241/bbb833c6db16461caaadbc628f6f9bb7.jpg',
    # 'img_66268a793071dd4f07eca241/9c9c6ffb42f741c686d97410d517fd17.jpg',
    # 'img_66268a793071dd4f07eca241/4ff6860c34864b4d95ae08c2f7023e6f.jpg',
    # 'img_66268a793071dd4f07eca241/ba076d42948d4c5d8aab392a4a2dcc49.jpg',
    # 'img_66268a5b3071dd4f07eca240/f592b7cefd8a4f81b2ba83b846826502.jpg',
    # 'img_66268a5b3071dd4f07eca240/d257d209c8f54e669f02a1afb23df855.jpg',
    # 'img_66268a5b3071dd4f07eca240/a0fb7455a45044739216e2cca627f473.jpg',
    # 'img_66268a5b3071dd4f07eca240/ed22d5f9a2094cf4bc5af6dc2f88b6c3.jpg',
    # 'img_66268a5b3071dd4f07eca240/536d1670615b4e1ba17a3f61890a099f.jpg',
    # 'img_66268a5b3071dd4f07eca240/4e0d93f70796430580ce7fe5d7a4e22a.jpg',
    # 'img_66268a5b3071dd4f07eca240/fdefb629609a409e8dc9125e82218d57.jpg',
    # 'img_66268a5b3071dd4f07eca240/f3936ca5de0e4887956dd88773502391.jpg',
    # 'img_66268a5b3071dd4f07eca240/f97e1b70d8f34083a4b093862f1d0cd8.jpg',
    # 'img_6639c419e1fb1f61863c235a/c81ada54b4e349d98786cd75da5a99c9.jpg',
    # 'img_6639c419e1fb1f61863c235a/5b152c37a9f6473dbf931080b7698c94.jpg',
    # 'img_6639c419e1fb1f61863c235a/09ce3f8aaf6347eaab1c9210f2b93ad4.jpg',
    # 'img_6639c419e1fb1f61863c235a/3c59b5c23153471b8a37cd39d23b0033.jpg',
    # 'img_6639c419e1fb1f61863c235a/c262ab01af0347d2a61ea4f74e4d0458.jpg',
    # 'img_6639c419e1fb1f61863c235a/1d932411e2414a64beb94f12428edb16.jpg',
    # 'img_6639c419e1fb1f61863c235a/461f3403acc547bfb98f02d8da5bfe1c.jpg',
    # 'img_6639c419e1fb1f61863c235a/f2f8c1a3adf24b01bc970c3961fb5217.jpg',
    # 'img_6639c419e1fb1f61863c235a/faa7af86a9d646efad1ed16fb9103f5b.jpg',
    # 'img_6639c419e1fb1f61863c235a/a20f7cd6b0034cee8c989c23991f8d64.jpg',
    # 'img_663c60469036c94341364415/d166aac619214437a4c9ee25fd763fcd.jpg',
    # 'img_663c60469036c94341364415/d966519fa0a14204b9ddf91092c54795.jpg',
    # 'img_663c60469036c94341364415/ec33dbe2240b434a92900213d048c3d5.jpg',
    # 'img_663c60469036c94341364415/259166c365d4401792112ddd71366559.jpg',
    # 'img_663c60469036c94341364415/e5a1c4398bf44bcab4675c0bb1e93046.jpg',
    # 'img_663c60469036c94341364415/9b7e6f36f1eb4bfb838a30630b9182eb.jpg',
    # 'img_663c60469036c94341364415/9cd7690101e745ce8ef6d3a8a008474d.jpg',
    # 'img_663c60469036c94341364415/7c728674787f4301b2f8f811a008112b.jpg',
    # 'img_663c60469036c94341364415/18351469262047a58a6e93baecd1105f.jpg',
    # 'img_663c60469036c94341364415/683a80b9f97e4f8583b386671872a405.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/057c166dcc8648e69cba69009a8535b4.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7868ef382f84429fb79047fba3978b67.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e1e599f026834cd0ae7f0a714123175d.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9cf26a02681f44f68b718762cd0f5494.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9d2658c034bb436f8e30aa5b2728dc77.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a9c987587751494798f2c5ee17181c58.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9d7bdfbce3394b2e95ebde7fbb684182.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1dfa3061d79142c683025c8267ee8499.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/160b82883aa04825bdb52ec08a7fac18.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d68fd34d74ba4c02b969065467e7ab56.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f42d61fcd2564975a4dd5ed3c965810b.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7104abb418784d879e5493614b10d812.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9e394126256f45ffb7f1fb180a31f481.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/331c7d16481545088b44dde0f3c92c07.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/c126a36d84fb46d592b2fdf122d2be7d.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/07fc9e732e2c422584e23364e221968f.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/34391d88fc3940deadff2e97fbea6e8c.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/379f25d4eef8409d8fc643ba6157abcd.jpg',
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4563bf6787974c829c8787406ab9e095.jpg'
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
        cv2.imwrite(f'orig/{img_names[img_idx]}_orig_result.jpg', img)
        print(f'下载图片耗时{time.time() - now:.2f}s')
        # img = e2c(img, face_w=int(img.shape[0] / 3))
        now = time.time()
        (height, width, depth) = img.shape
        equi_img = np.transpose(img, (2, 0, 1))
        equi_img = torch.tensor(equi_img)
        pers_should_height = height / 4
        pers_should_width = width / 4
        pers_height = 480
        pers_width = 640
        pitch = math.radians(0)
        fov_deg = 90.0
        rectangles = []
        img_PIL = Image.fromarray(img[..., ::-1])  # 转成 PIL 格式
        draw = ImageDraw.Draw(img_PIL)  # 创建绘制对象
        # 俯仰角从0到30度，每次转动5度
        for j in range(7):
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
                pers_height = 480
                pers_width = 640
                pers_img = equi2pers(
                    equi=equi_img,
                    rots=rots,
                    height=pers_height,
                    width=pers_width,
                    fov_x=fov_deg,
                    mode="bilinear",
                )
                cube_result = np.ascontiguousarray(np.transpose(pers_img, (1, 2, 0)))
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
                cv2.imwrite(f'{dir}/原图/origin_img_{idx}.jpg', pers_image.pers_img)
                for cls, box, conf in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
                    cls_np = int(cls.cpu().detach().numpy().item())
                    if cls_np not in name_dict.keys():
                        continue
                    box_np = box.cpu().detach().numpy().squeeze()
                    conf_np = conf.cpu().detach().numpy().item()
                    p1 = (box_np[0], box_np[1])
                    p2 = (box_np[2], box_np[3])
                    factor = 1 / 250
                    if (p2[0] - p1[0]) * (p2[1] - p1[1]) / (pers_width * pers_height) < factor:
                        continue
                    left_top = screen_to_equirectangular(box_np[0], box_np[1], pers_width, pers_height, fov_deg,
                                                         math.degrees(pers_image.yaw), math.degrees(pers_image.pitch),
                                                         width, height)
                    right_bottom = screen_to_equirectangular(box_np[2], box_np[3], pers_width, pers_height, fov_deg,
                                                             math.degrees(pers_image.yaw),
                                                             math.degrees(pers_image.pitch), width, height)
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
                    font = ImageFont.truetype(font='wqy-zenhei.ttc',
                                              size=40)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
                    orig_draw.rectangle(((p1[0], p1[1] - 50), (p1[0] + 300, p1[1])),
                                        fill=(255, 0, 0), )
                    name = f'{name_dict[cls_np]} {conf_np:.2f}'
                    orig_draw.text(xy=(p1[0], p1[1] - font.size - 10), text=name, font=font,
                                   fill=(255, 255, 255))
                    # cv2.imshow('img', cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR))
                    # cv2.waitKey(0)
                    r = Rectangle(left_top, right_bottom, cls_np, conf_np, pers_image.pitch, pers_image.yaw)
                    r.img_idx = idx
                    r.pers_p1 = [box_np[0], box_np[1]]
                    r.pers_p2 = [box_np[2], box_np[3]]
                    rectangles.append(r)
                print(f'保存标注图第{idx}张')
                cv2.imwrite(f'{dir}/标注图/labeled_img_{idx}.jpg',
                            cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR))
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
        cv2.imwrite(f'det/{img_names[img_idx]}_det_result.jpg', img)
        cv2.imwrite(f'det_result.jpg', img)

        # seg_img = requests.get(image_url + '?x-oss-process=image/resize,h_1024,m_lfit')
        # seg_img_arr = np.array(Image.open(io.BytesIO(seg_img.content)))
        # mask = predict_result(inferencer(seg_img_arr, show=False)['predictions'])
        # mask.save(f'seg/{img_names[img_idx]}_seg_result.jpg')

        print(f'总耗时:{time.time() - begin}秒')
