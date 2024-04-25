import io
import math
import urllib
import io
import cv2
import numpy as np
import os
import requests
from equilib import equi2cube, cube2equi


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
    return _u * W, _v * H

if __name__ == '__main__':
    # res = requests.get(
    #     'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3702d9e2dc2143c89ea98510195047e9.jpg')
    # Run equi2pers

    #
    # img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.imread('/home/lixiang/PycharmProjects/ultralytics/conf_0.1/cube_result_83.jpg')
    img = cv2.resize(img, (720, 1440))
    cv2.imshow('cube', cv2.resize(img, (1440, 720)))
    cv2.waitKey(0)

    # img = e2c(img, face_w=int(img.shape[0] / 3))
    (width, height, depth) = img.shape
    pic = np.zeros((width, height, depth))
    cut_width = int(width / 4)
    cut_height = int(height / 3)
    equi_img = np.transpose(img, (2, 0, 1))
    # rotations
    rots = {
        'roll': 0.,
        'pitch': 0,  # rotate vertical
        'yaw': 0,  # rotate horizontal
    }

    # Run equi2pers
    cube_img = equi2cube(
        equi=equi_img,
        rots=rots,
        w_face=cut_width,
        cube_format='dice'
    )
    point = (cut_width + 100, 100)
    transpose = np.transpose(cube_img, (1, 2, 0))
    transpose = np.ascontiguousarray(transpose)
    cv2.circle(transpose, point, 10, color=(255, 20, 0), thickness=-1)
    cv2.imshow('cube', transpose)
    cv2.waitKey(0)
    cube_img = np.transpose(transpose, (2, 0, 1))
    result = cube2equi(cube_img, 'dice', height=height, width=width)
    result = np.transpose(result, (1, 2, 0))
    (x, y) = map_cube(100, 100, 'top', cut_width, 1440, 720)
    cv2.imshow('cube', result)
    cv2.waitKey(0)
    result = np.ascontiguousarray(result)
    cv2.circle(result, (int(x), int(y)), 10, color=(0, 0, 255), thickness=-1)
    cv2.imshow('cube', result)
    cv2.waitKey(0)
    result = np.transpose(result, (1, 0, 2))
    cube_img = np.ascontiguousarray(np.transpose(cube_img, (2, 1, 0)))
    (cube_width, cube_height, _) = transpose.shape
    face_width = int(cube_height / 4)
    face_height = int(cube_width / 3)
    for i in range(3):
        for j in range(4):
            pic = transpose[i * face_width: (i + 1) * face_width, j * face_height: (j + 1) * face_height, :]
            cv2.imshow('cube', pic)
            cv2.waitKey(0)
    # img = c2e(img, 360, 720)
