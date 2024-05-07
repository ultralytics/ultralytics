import numpy as np


def viewport_to_camera_coordinates(x, y, width, height):
    """
    将视口坐标转换为相机坐标系中的坐标
    """
    # 将点坐标归一化到范围 [-1, 1]
    x_normalized = (2 * x / width) - 1
    y_normalized = 1 - (2 * y / height)  # 上下翻转

    return np.array([x_normalized, y_normalized, -1])  # z轴朝内


def camera_to_world_coordinates(camera_coordinates, camera_position, rotation_matrix):
    """
    将相机坐标转换为世界坐标系中的坐标
    """
    # 将相机坐标转换为世界坐标系中的坐标
    world_coordinates = np.dot(rotation_matrix, camera_coordinates) + camera_position

    return world_coordinates


# 示例：假设屏幕尺寸为5760x2880，点坐标为(2880, 1440)，相机位置为(0, 0, 0)，相机朝向为0度（即x轴正方向），俯仰角为30度
viewport_width = 5760
viewport_height = 2880
x = 2880
y = 1440
camera_position = np.array([0, 0, 0])
yaw = np.deg2rad(0)
pitch = np.deg2rad(30)

# 将视口坐标转换为相机坐标系中的坐标
camera_coordinates = viewport_to_camera_coordinates(x, y, viewport_width, viewport_height)

# 计算相机朝向的旋转矩阵
yaw_rotation_matrix = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
])

pitch_rotation_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(pitch), -np.sin(pitch)],
    [0, np.sin(pitch), np.cos(pitch)]
])

# 将相机坐标转换为世界坐标系中的坐标
world_coordinates = camera_to_world_coordinates(camera_coordinates, camera_position,
                                                np.dot(pitch_rotation_matrix, yaw_rotation_matrix))

print("世界坐标系中的坐标:", world_coordinates)
