import io
import math
import os.path
import time

import cv2
import numpy as np
import requests
import torch
from equilib import cube2equi, equi2cube
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO


def cubemap_to_equirectangular(u, v, face_size, face_index, equirectangular_width, equirectangular_height):
    # Convert u, v to range [-1, 1]
    u = (2 * u / face_size) - 1
    v = (2 * v / face_size) - 1

    # Determine face and adjust u, v accordingly
    if face_index == 0:  # Positive X face
        u, v = 1, v
    elif face_index == 1:  # Negative X face
        u, v = -1, v
    elif face_index == 2:  # Positive Y face
        u, v = u, 1
    elif face_index == 3:  # Negative Y face
        u, v = u, -1
    elif face_index == 4:  # Positive Z face
        u, v = u, -v
    elif face_index == 5:  # Negative Z face
        u, v = -u, v
    else:
        raise ValueError("Invalid face index")

    # Convert (u, v) to spherical coordinates
    x = u
    y = v
    z = 1
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)

    # Convert spherical coordinates to latitude and longitude
    latitude = math.degrees(theta) - 90
    longitude = math.degrees(phi)

    # Convert latitude and longitude to equirectangular coordinates
    equirectangular_x = (longitude + 180) * (equirectangular_width / 360)
    equirectangular_y = (90 - latitude) * (equirectangular_height / 180)

    return equirectangular_x, equirectangular_y


def get_theta_phi(_x, _y, _z):
    dv = math.sqrt(_x * _x + _y * _y + _z * _z)
    x = _x / dv
    y = _y / dv
    z = _z / dv
    theta = math.atan2(y, x)
    phi = math.asin(z)
    return theta, phi


# x,y position in cubemap
# cw  cube width
# W,H size of equirectangular image
def map_cube(x, y, side, cw, W, H):
    u = 2 * (float(x) / cw - 0.5)
    v = 2 * (float(y) / cw - 0.5)

    if side == "front":
        theta, phi = get_theta_phi(1, u, v)
    elif side == "right":
        theta, phi = get_theta_phi(-u, 1, v)
    elif side == "left":
        theta, phi = get_theta_phi(u, -1, v)
    elif side == "back":
        theta, phi = get_theta_phi(-1, -u, v)
    elif side == "bottom":
        theta, phi = get_theta_phi(-v, u, 1)
    elif side == "top":
        theta, phi = get_theta_phi(v, u, -1)

    _u = 0.5 + 0.5 * (theta / math.pi)
    _v = 0.5 + (phi / math.pi)
    return int(_u * W), int(_v * H)


# Load a model
model = YOLO("runs/detect/train33/weights/best.pt")  # pretrained YOLOv8n model
# Run batched inference on a list of images
images = [
    #     # 'images/3702d9e2dc2143c89ea98510195047e9.jpg',
    #     # '/home/lixiang/PycharmProjects/city-component-recognition/ba0ce43b60c74f3fa3d9c6edddd9a60f.jpg'
    # 'images/18c6c2729405445b98469205eb7aa9c2.jpg',
    # 'images/3e985529c9aa4db9b62a40c09287a207.jpg',
    # 'images/8f13444d270c4a268187b4925acf9f5c.jpg',
    # 'images/68504d2c780d4b1ea183f54a48f354a4.jpg',
    # 'images/87132c70129f494b9d05af880bce715d.jpg',
    # 'images/d8cde156911a472296f3ec5b3209c5b4.jpg',
    # 'images/382cfcd217714ad6b827562d32d047d8.jpg',
    # 'images/8707df279b904b2dad570f7c57b3f4d0.jpg',
    # 'images/63ba5b0f080f4502848187211ee5b0c7.jpg',
    # 'images/d9ee261e302d4e01ba17329960f75a21.jpg',
    # 'images/048a5c4d378b400b8b66fbc7b552276a.jpg',
    # 'images/0544b0760ead4bb5a93400a643d6c3d7.jpg',
    # 'images/6948dd9abc4b4e8e93bd96d4ce663508.jpg',
    # 'images/d6b3994ff80f4653bac88e85011aa6c0.jpg',
    # 'images/f32beb5376644020858c6a55696b0f16.jpg',
    # 'images/ad6bc1a866fa4028a2c5bfbd7e97cb81.jpg',
    # 'images/68191aab79614b00b28dff1ccac3d057.jpg',
    # 'images/ac9b7ed8a4b346c38c56aa701ce3b2a7.jpg',
    # 'images/fd8b6ac7d6b4452ea67331db36c8f776.jpg',
    # 'images/9298763965df4a4c842e5c7d7e4313ea.jpg',
    # 'images/46139f0acd2b46cf816086957c96600b.jpg',
    # 'images/4e7e1f23f7fa4d1d8c6ca437eea7714c.jpg',
    # 'images/d44aa80de24c4d8984c1ff1f4da8d9b9.jpg',
    # 'images/721262c0cec946588e773770ca487f84.jpg',
    # 'images/6148b013b39843f98787ac457fe6a1cc.jpg',
    # 'images/52cc692a0a6b474eab914416eba1e5b0.jpg',
    # 'images/84792606056b47829862a4b04db2f24a.jpg',
    # 'images/ab252c1dadb04ffabadc4e3d33041db6.jpg',
    # 'images/f323db02602743849afd76ddb7b078b4.jpg',
    # 'images/42934bb918f343a388179cef63c247ef.jpg',
    # 'images/7494954223e147ef9eb82260b4177e6c.jpg',
    # 'images/389a95780de840fab29e8189bab34c05.jpg',
    # 'images/e16970f3849d4f579b0eca031deaca7e.jpg',
    # 'images/5bde7424ea66406e93a342a329053aa9.jpg',
    # 'images/fd3eb02cce0f4764bd701b542a000ab4.jpg',
    # 'images/f39892f5d4f64e4dadec31772ad6f450.jpg',
    # 'images/2a8291871cab49a482d6a43a50543a16.jpg',
    # 'images/b18606efd20f4482b5eeb0d9fd60ba97.jpg',
    # 'images/6861b9fa47c54205b8797cf90605d74f.jpg',
    # 'images/a597cf9d9a8e477892a16c216d77414d.jpg',
    # 'images/f9d4862809f844fdbb3100567524573d.jpg',
    # 'images/3d15c220c2be4e0084b152be6fe4e04b.jpg',
    # 'images/6a1953facd544a19acb558a1e3d29658.jpg',
    # 'images/11cef6c1a2cf4e3fab473bd29dd641c2.jpg',
    # 'images/78c30b1b21724a9e87ecff127d7cf159.jpg',
    # 'images/e9b4254e7de846a285170527fbb2d615.jpg',
    # 'images/979913364bfa4496a6cd680307ccb039.jpg',
    # 'images/c7be016c337845d79d0dda46e475ab02.jpg',
    # 'images/a08f2a9a927a45bb80d3a64783b14152.jpg',
    # 'images/c8fdcb8a55b84b4ab8f98dc45b43a405.jpg',
    # 'images/5d7c8ed4c2c44d92950e79351aae1dba.jpg',
    # 'images/50067d3313c84f329ac8e15de483c727.jpg',
    # 'images/de2e6c7288b04764a4ea3668afd8060b.jpg',
    # 'images/b7d41a9fc88d4e8f96fd94194917d09c.jpg',
    # 'images/265c889be4274fd0866a7bc6d57492f8.jpg',
    # 'images/b14c5d35cdec4300ba3bfba2189af6a2.jpg',
    # 'images/fd24bbbf9c83404e963db160688206c0.jpg',
    # 'images/cd428a3333284712ac0a5268e487cf52.jpg',
    # 'images/11f0365fc8464dfbbd6c7d565c915140.jpg',
    # 'images/ddbc6e75046b408d84f427620f5f2cf1.jpg',
    # 'images/b1548736e0c742d1aecda723b3dc564f.jpg',
    # 'images/ac4ad91f1d2e4ab58b142206b6c8cef8.jpg',
    # 'images/1b9bc6ec09924c7aaa0a22c46aa36c30.jpg',
    # 'images/f0b1f8e0ecb04c46ad4981c3949723ba.jpg',
    # 'images/5e36ac3647444cabb822ebb4fb7386bd.jpg',
    # 'images/10ced56e9cea493db94586517b49b07e.jpg',
    # 'images/63481d075f42463389092da85bbbd489.jpg',
    # 'images/83312b42a1e84eb4a01c7472abf784d8.jpg',
    # 'images/d66b65c08e864a63923f28bfe1b72a09.jpg',
    # 'images/c0f4d1b66578409bbca8180f4dd1b4bb.jpg',
    # 'images/0553429a112d497286592ecb5295cb66.jpg',
    # 'images/cdc782487a74428dbfb09c998a525f8f.jpg',
    # 'images/f74d9ef1ed41467eb83e7f856eead203.jpg',
    # 'images/aaeba3bd972d42d48a90a10b7a07a27d.jpg',
    # 'images/88d7bf63c70a425d936b324f9ecc5be1.jpg',
    # 'images/c86ae80e4be543d69b2605c5e5fed85b.jpg',
    # 'images/47fc69bdabb44d068fe2f480d4306275.jpg',
    # 'images/ba9926ee83ed4e629a4d9d6655c70c0f.jpg',
    # 'images/9f9ee6b7f01149778dd6ce4b7e9cb874.jpg',
    # 'images/a0bca8e4348e4eacbdffc21122c3f5b8.jpg',
    # 'images/e2fafadc1fa84b488641c11bb559992b.jpg',
    # 'images/350913474c31431ca3f73d774ca89e91.jpg',
    # 'images/91611c5cfce749c697a5c3b990613296.jpg',
    # 'images/a1e1d6c5131747739e490758c9640540.jpg',
    # 'images/4b29091618c14fea9ec93eb78910cc53.jpg',
    # 'images/52d97ed40a3442e28a77ae08b4a4d89e.jpg',
    # 'images/a5105645a5fa4e7bae997ca28c3135ea.jpg',
    # 'images/85e66ab6d29f43a682fd9a36fa583465.jpg',
    # 'images/53cb6f08c1054710b63df4a5ea907cd5.jpg',
    # 'images/2e0d69baaffb4b57ac1d774214c098be.jpg',
    # 'images/f09f864e14bc4d6e922ec825695e8f68.jpg',
    # 'images/0d95413d4f9a4fe6805aacfa49a8d407.jpg',
    # 'images/cfbff1d2e4f84bc391844caff5c57565.jpg',
    # 'images/c53c0cf1f6e54165804871caa0285767.jpg',
    # 'images/052d09b5aa1648aa9b3aa8945c2045d1.jpg',
    # 'images/fd3dc481f32f44fcb402b0f4ed6805a1.jpg',
    # 'images/66504c58501b46c2bbb6875418dd3c4e.jpg',
    # 'images/ad29c01c6baa40839f9e53140e312a7e.jpg',
    # 'images/aea8e0f0fec448758477f1020137a946.jpg',
    # 'images/4f55e3afe0f8493fb0c622a6b82971f8.jpg',
    # 'images/c267067bda97459d8c2e883f936b8163.jpg',
    # 'images/7ee896960ab948a1a05c61ca8acd00d4.jpg',
    # 'images/61187707fdef421dbed63c30e8c9869f.jpg',
    # 'images/46a9bae9a42b4315b17fbce2954a01d0.jpg',
    # 'images/d1051ff11ac4495690618924716ff249.jpg',
    # 'images/3a0a90b0477e4b38a82248f915d62aeb.jpg',
    # 'images/06a7df070916435d83a72c1c6982fb44.jpg',
    # 'images/f409281dca06440c99f42b8c61c35937.jpg',
    # 'images/bbf5fa07eeab47348b91824f0ddd2e67.jpg',
    # 'images/ae3c8ceef40c4602a846086d9ee62f17.jpg',
    # 'images/a86d5e16644a409d928b8cd490050da1.jpg',
    # 'images/0e90db3d2b834ebf9ef0e709b7dcb866.jpg',
    # 'images/20f71c6d9a2e4dae80461645f025ce58.jpg',
    # 'images/d5edd51d9f6a4ac2888502fa0bfc3517.jpg',
    # 'images/0cbe5dfa0e034bcbb23dd0d1c6ac6a9f.jpg',
    # 'images/8d47b9a3d73449878bbe14ddfc3949a8.jpg',
    # 'images/e51fd8d296fc4cd4b9bc86bc0582ac6c.jpg',
    # 'images/7f88afe9d9a64d7293b6161741cc2b25.jpg',
    # 'images/ea7a00da5dbb4a3e916e40eda72af2ea.jpg',
    # 'images/e704e4694bfd4fab97c312fa1f85dfeb.jpg',
    # 'images/bd933e0b8e824586a47e325393bb94b7.jpg',
    # 'images/537f6237d15b4afba4cae6ec42875141.jpg',
    # 'images/cdda1d2bb3a2477ca0afac4dc33c5415.jpg',
    # 'images/a7f3dc48b0454773b63ada4b23ae8cff.jpg',
    # 'images/8bde3bc7ad694371bfe8afeb23c6bf31.jpg',
    # 'images/8b71dbde4bee4d3e821f832de9278c87.jpg',
    # 'images/b1c4dd17438046f2a8c28a1bb02a1910.jpg',
    # 'images/0c5e612f129544df89b33711a6a87b9e.jpg',
    # 'images/0ff6395660af4fd1a16e9b4ca8324401.jpg',
    # 'images/057a9104f5244c58a32b990c5a3fafe8.jpg',
    # 'images/0b83c506acb94516b8fb25a01fb57ee7.jpg',
    # 'images/a45cf022c5664939a8e520ecd6fa7c34.jpg',
    # 'images/151a956591fd427facbf38702ba203ed.jpg',
    # 'images/6939879158f94de1bc23af8a91a52c62.jpg',
    # 'images/d2c0eb182f8a4c21b5edc4a8684c337d.jpg',
    # 'images/7715a7e4cf90409b9a51d27f968dc57d.jpg',
    # 'images/4272e88b90fe487bb443a649bf0a663d.jpg',
    # 'images/96b2dbee15584489bc430242eb5e8e61.jpg',
    # 'images/9f65a2d3d62f46e89de4d741ed1f29bc.jpg',
    # 'images/74438138d10a4d5a8e2d0b49c341463d.jpg',
    # 'images/e26ab0be80224de688dead7cbfb811fe.jpg',
    # 'images/9a0f214ab2cf4a50a3ee6ae0ab93c9c5.jpg',
    # 'images/1e4a1af8d4e4427ab35af7ab848f782e.jpg',
    # 'images/dccca12fc925481b83e4e7c538587e7c.jpg',
    # 'images/ce5c38a074694fa28210e5d631fe4fb0.jpg',
    # 'images/4eaa6f38f4c541ae8b825058255bd998.jpg',
    # 'images/0b3cc9714e2e42db9261f762a30f4817.jpg',
    # 'images/0fa0e1b94f10434a8e7d13e22a36ed01.jpg',
    # 'images/2151def651bf43fc931f2e724b180e34.jpg',
    # 'images/2d633c98fbb5464d93d06605f09a9c08.jpg',
    # 'images/d87e4beece88403fb374ffbbe11f67bd.jpg',
    # 'images/2630e1cd57594dc5967f43dc2c362f33.jpg',
    # 'images/a4c045ea5a1a4d068023b4bdca97f48d.jpg',
    # 'images/9e0dc69b8fe24429a9d7629bd7ed14f1.jpg',
    # 'images/6bce372247014fb380e9487ee8c8942b.jpg',
    # 'images/6354a53a607a4090819b8cb72e720c88.jpg',
    # 'images/115ecd0205174519b5c27dcf359aaf0c.jpg',
    # 'images/6eb981cd2e86484499d59907663b80d5.jpg',
    # 'images/c50ac79cc7d24904b8d29332db3f531d.jpg',
    # 'images/4cf8880410524911806ac8d8f4b04900.jpg',
    # 'images/d3e3b92c09974543a8184bc0a0555a12.jpg',
    # 'images/fb306ec1fe3844dd802d5ef4cf051d7d.jpg',
    # 'images/840ba63bec194e0aa71a0cbf0d11e7e2.jpg',
    # 'images/ebcef4d99a91433194e561551aba2419.jpg',
    # 'images/eb4627eee39241eeb808026691ad69e0.jpg',
    # 'images/92583406f29245a3a155f34f6de11a30.jpg',
    # 'images/3a68e30fc8344f9ebebecedcecdfaee8.jpg',
    # 'images/f3c1b0ffc5b049cda49e9004a432a57d.jpg',
    # 'images/b78b00f2ba33480d8ad574e892efeba4.jpg',
    # 'images/3f2679dc6b49401b8d20d968611b5f9d.jpg',
    # 'images/c4b1b2b69438426e87efffedc8852d62.jpg',
    # 'images/3dd35efaa869466980eb3771c2906ad9.jpg',
    # 'images/91bd78cd8b0341d88b9b4dbfd2d84266.jpg',
    # 'images/f42e81e7ef4b45b09b9ec05c01c7de1b.jpg',
    # 'images/3702d9e2dc2143c89ea98510195047e9.jpg',
    # 'images/0c55c621d64e4e0cb75ffe9499e41429.jpg',
    # 'images/f295253a12b542c0980dbe182f803ec8.jpg',
    # 'images/d31518199c994771b22b447262b39e52.jpg',
    # 'images/022d220fce1d4af591a84fbcbe001406.jpg',
    # 'images/7422ab14a00e45b3869624801b4b9ee0.jpg',
    # 'images/bb418f0ee71e41f5b058474c7a7efa2d.jpg',
    # 'images/52b4ce1d3bd44ec3bf8e8346d217c452.jpg',
    # 'images/2be3d91e215648fd81a9ddbafca84a9e.jpg',
    # 'images/f32b1ad5cec44fecbe38bbaf43ec38a5.jpg',
    # 'images/93451d4d97e646908e4d0bda46a65021.jpg',
    # 'images/e25d8a6a1d47440d9a69d81f5456f15e.jpg',
    # 'images/8c5a11fbe0cb463486f2662e17bd700b.jpg',
    # 'images/c2b44c3bf3e24d8d8024a7ce8e71d1ef.jpg',
    # 'images/b76cf81322924e3ca739ecfec8a141aa.jpg',
    # 'images/47aa56674f4e4cb4be3b608e0872fcf5.jpg',
    # 'images/e0d48f058faf434b8863cd54ae4e888e.jpg',
    # 'images/48c1a67280ce4b6faa772c677ab0bb65.jpg',
    # 'images/d931f7f62ac842b2a0f8b068b9bc4572.jpg',
    # 'images/c1bb53aa69b3443abc96f44965fcbe60.jpg',
    # 'images/66623bfb05974c2dbe84537a3144bc99.jpg',
    # 'images/f507a661b38b45468dff71cce9228f06.jpg',
    # 'images/fd57c33b224d46678c09cb747b2d78b4.jpg',
    # 'images/34c3c8325e764833848c62559a21aca5.jpg',
    # 'images/5a049668850e4413b742fb2cab2fca9d.jpg',
    # 'images/e6af75ffc69b449b8af938f8ee846580.jpg',
    # 'images/96939306a3664f14ab4b9aee49acde32.jpg',
    # 'images/014756f7b05c413e8913971140561db8.jpg',
    # 'images/17c0a8d979cb46678d8b4080e1af6be5.jpg',
    # 'images/d7ef86791ccb48d397060001b52ee271.jpg',
    # 'images/6dc88fcd1ef74c049b965bd4f693246b.jpg',
    # 'images/5a13d86145f64545887cebe67dae4b75.jpg',
    # 'images/758d996c3f304e83be84f53282121e10.jpg',
    # 'images/b781556f92b742f1b791cde71d13062b.jpg',
    # 'images/251b4b1417be4f83b4f4642b40a9ddc8.jpg',
    # 'images/5a26bc7b7fc14d3caae7c78df7b937e6.jpg',
    # 'images/d1931817dbed4756bcab0f4abed98cc9.jpg',
    # 'images/eec03a6ede9744d2bd854d3b71fddf8b.jpg',
    # 'images/d0ba4a57a81345de8bcbcce839ebb49e.jpg',
    # 'images/5c24a12e5b624d85b112b4ff016a54d5.jpg',
    # 'images/23e117af93d341f1b2d4dd6aa24d47c1.jpg',
    # 'images/c380e65803bc441580ed31bc82f6393e.jpg',
    # 'images/ec1ef8398ae343f4974b01287d0579d0.jpg',
    # 'images/0293c5aa5eb447938f80a12a2f981d7d.jpg',
    # 'images/e9cf51a229a34b348de91bbbb9cc4475.jpg',
    # 'images/fd198069d46842199e8e834b720d7dd8.jpg',
    # 'images/d99bde8d94e14ab89a0af1f9b6430ec6.jpg',
    # 'images/22ecd384672244a6b5806a8b0e9fc341.jpg',
    # 'images/a43180c23ef247cebbb5cf410faa9cf2.jpg',
    # 'images/ebbc09e4a3b34cb1b30a4febf54125f4.jpg',
    # 'images/a762f1723426490caec339d7a8d4dfd0.jpg',
    # 'images/4f128405a7e84f93baa69ca09384191d.jpg',
    # 'images/78f1a561116d424486de9deec3211157.jpg',
    # 'images/040fe524284344ac96f1c058dcb02dfc.jpg',
    # 'images/48f70ef2018745a2a8a15ecd279091b0.jpg',
    # 'images/6d3ed00bd892443f84421594eab92256.jpg',
    # 'images/9df5ad60d2a441f4b39260093acd61a1.jpg',
    # 'images/bbc5461821354c219bf8bc7b79d59c6f.jpg',
    # 'images/2bff4274433c4899b63af328cef08b41.jpg',
    # 'images/2f0e14246e934a62b7cca822ddc192ac.jpg',
    # 'images/d9f884d71d4c4b07bea774da8430de14.jpg',
    # 'images/aec62255d63d41a9b2e1c3f3cad11149.jpg',
    # 'images/deafbacb821e4982b8ed3d5e19285ad9.jpg',
    # 'images/700122c02be0424db3ff9b35c372dc45.jpg',
    # 'images/050f21a6dce4449ba583377b7b0d03ab.jpg',
    # 'images/c0fce6b627f54ccbb4c391479b3095bf.jpg',
    # 'images/9d48d7f8e21e404c8989d18379fb8001.jpg',
    # 'images/9f3130a506fc40b78e72d926e2979436.jpg',
    # 'images/f3f6266294b04ad6bb5c19a5fd563f62.jpg',
    # 'images/5231fd8533dd4b4b8f06f054204a8e7f.jpg',
    # 'images/9b5441c04c394ca0ad6b1438bbed3297.jpg',
    # 'images/e6da3b147d894418ab5312914ad56e1a.jpg',
    # 'images/86496e1980c34df8923e4e6f23dada3e.jpg',
    #     'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/02df7f0d12e5478d8ffef77a40861034.jpg'
]

images = [
    "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/52f3a0273dfb466e85e6bf59ade05c2e.jpg",
]

img_names = [x.split("/")[-1].split(".")[0] for x in images]

name_dict = {
    0: "路面破损",
    # 1: '沿街晾晒',
    # 2: '垃圾满冒',
    3: "乱扔垃圾",
    # 4: '垃圾正常盛放',
}

confs = [0.1]

for idx_conf in confs:
    if not os.path.exists(f"conf_{idx_conf}"):
        os.mkdir(f"conf_{idx_conf}")
    for img_idx, image_url in enumerate(images):
        now = time.time()
        res = requests.get(image_url)
        img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.imread(image_url)
        if img is None:
            continue
        # img = cv2.imread(image_url)
        print(f"下载图片耗时{time.time() - now:.2f}s")
        # img = e2c(img, face_w=int(img.shape[0] / 3))
        now = time.time()
        (height, width, depth) = img.shape
        pic = np.zeros((height, width, depth))
        cut_width = int(width / 4)
        cut_height = int(height / 3)
        equi_img = np.transpose(img, (2, 0, 1))
        # rotations
        rots = {
            "roll": 0.0,
            "pitch": 0,  # rotate vertical
            "yaw": 0,  # rotate horizontal
        }
        equi_img = torch.tensor(equi_img)
        # Run equi2pers
        cube_img = equi2cube(equi=equi_img, rots=rots, w_face=cut_width, cube_format="dice")
        print(f"图片处理耗时{time.time() - now}s")
        cube_result = np.ascontiguousarray(np.transpose(cube_img, (1, 2, 0)))
        face_width = int(cube_result.shape[1] / 4)
        face_height = int(cube_result.shape[0] / 3)
        top = cube_result[0 * face_height : 1 * face_height, face_width : 2 * face_width]
        bottom = cube_result[2 * face_height : 3 * face_height, face_width : 2 * face_width]
        left = cube_result[face_height : 2 * face_height, 0:face_width]
        front = cube_result[face_height : 2 * face_height, face_width : 2 * face_width]
        right = cube_result[face_height : 2 * face_height, 2 * face_width : 3 * face_width]
        back = cube_result[face_height : 2 * face_height, 3 * face_width : 4 * face_width]
        now = time.time()
        results = model.predict(
            [top, bottom, left, front, right, back], conf=idx_conf, imgsz=640
        )  # return a list of Results objects
        print(f"图片推理耗时{time.time() - now}s")
        idx_face_dict = {0: "top", 1: "bottom", 2: "left", 3: "front", 4: "right", 5: "back"}
        res = {}
        img_PIL = Image.fromarray(img[..., ::-1])  # 转成 PIL 格式
        draw = ImageDraw.Draw(img_PIL)  # 创建绘制对象
        for idx, result in enumerate(results):
            for cls, box, conf in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
                cls_np = int(cls.cpu().detach().numpy().item())
                box_np = box.cpu().detach().numpy().squeeze()
                if cls_np not in name_dict:
                    continue
                if cls_np not in res:
                    res[cls_np] = []
                conf_np = conf.cpu().detach().numpy().item()
                p1 = map_cube(box_np[0], box_np[1], idx_face_dict[idx], face_width, width, height)
                p2 = map_cube(box_np[2], box_np[3], idx_face_dict[idx], face_width, width, height)
                left_top = (p1[0] if p1[0] < p2[0] else p2[0], p1[1] if p1[1] < p2[1] else p2[1])
                right_bottom = (p2[0] if p2[0] > p1[0] else p1[0], p2[1] if p2[1] > p1[1] else p1[1])
                draw.rectangle(xy=(left_top, right_bottom), fill=None, outline="red", width=10)
                # cv2.rectangle(img=img, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=10)
                res[cls_np].append(
                    {"points": [{"x": p1[0], "y": p1[1]}, {"x": p2[0], "y": p2[1]}], "confidence": conf_np}
                )
                font = ImageFont.truetype(
                    font="uming.ttc", size=40
                )  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
                draw.rectangle(
                    ((left_top[0], left_top[1] - 50), (left_top[0] + 300, left_top[1])),
                    fill=(255, 0, 0),
                )
                name = f"{name_dict[cls_np]} {conf_np:.2f}"
                draw.text(xy=(left_top[0], left_top[1] - font.size - 10), text=name, font=font, fill=(255, 255, 255))
                img = cv2.cvtColor(
                    np.asarray(img_PIL), cv2.COLOR_RGB2BGR
                )  # 再转成 OpenCV 的格式，记住 OpenCV 中通道排布是 BGR

        # img_PIL.show()
        print(res)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.imwrite(f'cube_result_{img_idx}.jpg', img)
        cv2.imwrite(image_url.split("/")[-1], img)
