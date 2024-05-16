import numpy as np


def screen_to_equirectangular(x, y, screen_width, screen_height, fov, yaw, pitch, equi_width, equi_height):
    # 将屏幕坐标(x, y)转换到NDC坐标(-1 to 1)
    nx = (x / screen_width) * 2 - 1
    ny = (y / screen_height) * 2 - 1

    # FOV的一半的切线值，用于计算z坐标
    t = np.tan(np.radians(fov / 2))

    # 逆向计算出对应的方向向量
    direction = np.array([t * nx, t * ny, 1])
    direction = direction / np.linalg.norm(direction)

    # 建立旋转矩阵，考虑偏航角和俯仰角
    cos_yaw, sin_yaw = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cos_pitch, sin_pitch = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

    rotation_matrix = np.array([[cos_yaw, 0, -sin_yaw], [0, 1, 0], [sin_yaw, 0, cos_yaw]]) @ np.array(
        [[1, 0, 0], [0, cos_pitch, sin_pitch], [0, -sin_pitch, cos_pitch]]
    )

    # 将方向向量旋转到最终的方向
    final_dir = rotation_matrix @ direction

    # 计算球面坐标
    longitude = np.arctan2(final_dir[0], final_dir[2])
    latitude = np.arcsin(final_dir[1])

    # 转换为等距平面坐标
    ex = (longitude + np.pi) / (2 * np.pi) * equi_width
    ey = (latitude + np.pi / 2) / np.pi * equi_height

    return ex, ey


# 测试函数
screen_width = 1024
screen_height = 512
x, y = 512, 512  # 屏幕中心
fov = 90
yaw = 180  # 无偏航角
pitch = 0  # 无俯仰角

ex, ey = screen_to_equirectangular(x, y, screen_width, screen_height, fov, yaw, pitch, 5760, 2880)
print(f"Equirectangular coordinates: ({ex}, {ey})")
