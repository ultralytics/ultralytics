import os.path

import cv2
import numpy as np
import requests

imgs = {
    "662504a04dbc57656cfce1ed": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/926c8249e8454d2c83ba81f4b06b30c0.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4ae3658f9f794ca08b462c6fe6cc7390.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e3a4dd496cc142e5977fa841c1cdb588.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/748f05324f1444fd8f6b232300ccec6e.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/51f9d80eaa8746919ab777ba974e8678.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/cb3cdca3ddbe4805af66a4b88bb057d9.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/fe146e68a6c84c868243a01bec88e4c8.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/59117f01c8eb4579afbb04b6e9163925.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/013e92db2aa0445b9b7c606d53c2c923.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6ee72c73a2f44f689349f1e041ea233d.jpg"
    ],
    "662b005089d953346ed1c475": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/33b749b9bd2f40a4840e17d5254e16ed.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ce965bad47734fcbb026788bba50e1f7.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a6a88f1785fe4042b24338b565c1fd63.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f274bc26311c4ab28b2444082f2a8d49.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/72fadd4ee38a4ad392332aca03c85faa.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/06d54efd006e490d8aa5603260ccb4c2.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f85c77a28cba414dbd1ed9a5a24307d9.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0e737a627235479c8ff3a1a76100bb9f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0c2d58d8f45d4a4184d9df65d77a61f5.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/85a20c16d6ab41d18661271a44f88aed.jpg"
    ],
    "662680943071dd4f07eca23c": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/40664431b6574654bb77b9ebebfbbd55.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9a686709e7584f9d8ca10578590416c7.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/2afed2504eaa4f1290daafe0953591b0.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/395d151408804ae981d5d3a636d12fdd.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3980c02e40054692ae46c3ad6d55bbea.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/be5aba00f33f4814ad1f6cfd8f2b0ba2.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3f08a283daf74389b21743f91afdeb38.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0761f55764f248e586bca24dcfca41dd.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3071d1779de24408a740633e8aa16f64.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/be1c80cb99414663bdaa7500b74c8456.jpg"
    ],
    "663c600f9036c94341364414": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/dcb8e696cc0f4218a43ce5bb94aaebd0.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/59c57873ff1c46449017f7e58d0f0e1f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ba39f9338c6f43298a03d923833ec62b.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/8f855773ff664bf99d9a5d990c09cd08.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3211f225c2d44afe85f301f55b0c09d9.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0cec5eb87ce14537b666b09dcb3c8849.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4f13bf389b5d49d8b698b630be4b6e25.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d7fea3a81c0d4783b9f3dd3fb5c1ffee.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3d9ed423e76e4074b47fcfafd3af5bb3.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5b619abf3b29451f9d7850789f3577a2.jpg"
    ],
    "662680703071dd4f07eca23b": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/226134332a1943f2b7ddd8f93e94c439.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5dd5f033eb134b1ab3c40a082c28bc61.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/77f7cea57d564527832efb59017481f4.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3ccce744cc6e4bb8a90d60f54c3525de.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1f1c63f7dcbe43f7ba174688d2bfc01f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1f76e22065e9433fa63d9b3843e09f17.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ec2869c2311b4129a830489f91440249.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/08e00b6412c847ecb18c84b81449ec15.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/efa8564101ec4906843e8519c6d5f44f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/26d62670f5c84f1e92c0402b0e824130.jpg"
    ],
    "6639c3e6e1fb1f61863c2359": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1ff0ca6dd1354fe6ad651a3fb929356f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9f2a417c4e1b488a9f340c33f38f7eeb.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/2d8831de52814dc18909d788e05d438a.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1cc1beb7ae3f46a29b7a56cc1478d41e.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/12ca6d7291734bc2924d64f9097e4bef.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/afb4cb43e14941e7b074123c040f747b.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/8cac48fc625543839aa1bae0898d88a4.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5283352cbdf343bd8e4ce197cbbd084a.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7289a96720c34e3a83adf62867fc6615.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a0302eaf6dc84a84842970feb9cab50f.jpg"
    ],
    "66268a793071dd4f07eca241": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/71e86432ce6d480bafa2c43493ec4568.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/789a03d9970d433cbdfb62e24498454d.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9e6b4cce1bf84c40829a4cad796fe967.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/2107cc43c470409a9fcaa120e100476e.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6e0aeebd172640b88c72d3fc88f3fab9.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/eab9eed230cc484da5ea07a3d7cf2bef.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/bbb833c6db16461caaadbc628f6f9bb7.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9c9c6ffb42f741c686d97410d517fd17.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4ff6860c34864b4d95ae08c2f7023e6f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ba076d42948d4c5d8aab392a4a2dcc49.jpg"
    ],
    "66268a5b3071dd4f07eca240": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f592b7cefd8a4f81b2ba83b846826502.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d257d209c8f54e669f02a1afb23df855.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a0fb7455a45044739216e2cca627f473.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ed22d5f9a2094cf4bc5af6dc2f88b6c3.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/536d1670615b4e1ba17a3f61890a099f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4e0d93f70796430580ce7fe5d7a4e22a.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/fdefb629609a409e8dc9125e82218d57.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f3936ca5de0e4887956dd88773502391.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f97e1b70d8f34083a4b093862f1d0cd8.jpg"
    ],
    "6639c419e1fb1f61863c235a": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/c81ada54b4e349d98786cd75da5a99c9.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5b152c37a9f6473dbf931080b7698c94.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/09ce3f8aaf6347eaab1c9210f2b93ad4.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3c59b5c23153471b8a37cd39d23b0033.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/c262ab01af0347d2a61ea4f74e4d0458.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1d932411e2414a64beb94f12428edb16.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/461f3403acc547bfb98f02d8da5bfe1c.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f2f8c1a3adf24b01bc970c3961fb5217.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/faa7af86a9d646efad1ed16fb9103f5b.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a20f7cd6b0034cee8c989c23991f8d64.jpg"
    ],
    "663c60469036c94341364415": [
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d166aac619214437a4c9ee25fd763fcd.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d966519fa0a14204b9ddf91092c54795.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ec33dbe2240b434a92900213d048c3d5.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/259166c365d4401792112ddd71366559.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e5a1c4398bf44bcab4675c0bb1e93046.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9b7e6f36f1eb4bfb838a30630b9182eb.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9cd7690101e745ce8ef6d3a8a008474d.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7c728674787f4301b2f8f811a008112b.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/18351469262047a58a6e93baecd1105f.jpg",
        "https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/683a80b9f97e4f8583b386671872a405.jpg"
    ]
}

for id in imgs:
    path = f'img_{id}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    for image_url in imgs[id]:
        # res = requests.get(image_url)
        # img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        # cv2.imwrite(os.path.join(path, image_url.split('/')[-1]), img)
        # print(f'下载: {image_url}')
        print(os.path.join(path, image_url.split('/')[-1]))