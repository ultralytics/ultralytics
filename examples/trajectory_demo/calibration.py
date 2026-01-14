"""
calibration.py.

Homography矩阵计算工具（简化版）

使用流程：
1. 提供4个参考点的像素坐标和对应的世界坐标
2. 计算homography矩阵
3. 保存矩阵到JSON文件,供后续使用

用法：
python calibration.py --pixel-points "100,50 1800,80 1850,1000 120,1050" --world-points "0,0 12,0 12,8 0,8" --output calibration/
"""

import argparse
import json
import os

import cv2
import numpy as np


class HomographyCalibrator:
    def __init__(self, pixel_points, world_points, output_dir="calibration"):
        """初始化标定工具.

        参数：
        - pixel_points: [(px1,py1), (px2,py2), (px3,py3), (px4,py4)] 像素坐标列表
        - world_points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 世界坐标列表
        - output_dir: 输出目录（保存矩阵）
        """
        if len(pixel_points) != 4 or len(world_points) != 4:
            raise ValueError("必须提供恰好4个像素坐标和4个世界坐标")

        self.pixel_points = pixel_points
        self.world_points = world_points
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_homography(self):
        """计算homography矩阵."""
        # 转换为numpy数组
        src_pts = np.float32(self.pixel_points)
        dst_pts = np.float32(self.world_points)

        # 计算homography矩阵
        H, _ = cv2.findHomography(src_pts, dst_pts)

        if H is None:
            print("❌ 计算homography矩阵失败")
            return None

        print("\n✓ Homography矩阵计算成功!")
        print("\n矩阵内容:")
        print(H)

        return H

    def save_homography(self, H, video_name="calibration"):
        """保存homography矩阵到JSON文件."""
        if H is None:
            return None

        # 转换矩阵为列表（便于JSON序列化）
        H_list = H.tolist()

        output_file = os.path.join(self.output_dir, f"{video_name}_homography.json")

        data = {
            "video_name": video_name,
            "homography_matrix": H_list,
            "pixel_points": self.pixel_points,
            "world_points": self.world_points,
            "notes": "Use this matrix to convert pixel coordinates to world coordinates",
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Homography矩阵已保存:{output_file}")
        return output_file

    def test_transform(self, H):
        """测试变换效果."""
        if H is None:
            return

        print("\n【变换效果测试】")
        print("-" * 50)

        for i, (px_pt, w_pt) in enumerate(zip(self.pixel_points, self.world_points)):
            # 手动进行透视变换
            pixel_coord = np.array([[[px_pt[0], px_pt[1]]]], dtype=np.float32)
            world_coord = cv2.perspectiveTransform(pixel_coord, H)
            transformed_x, transformed_y = world_coord[0][0]

            error_x = abs(transformed_x - w_pt[0])
            error_y = abs(transformed_y - w_pt[1])

            print(f"点 {i + 1}:")
            print(f"  像素坐标: {px_pt}")
            print(f"  输入的世界坐标: ({w_pt[0]:.2f}, {w_pt[1]:.2f})")
            print(f"  变换后坐标: ({transformed_x:.2f}, {transformed_y:.2f})")
            print(f"  误差: ({error_x:.4f}, {error_y:.4f}) 米")
            print()


def parse_coordinates(coord_str):
    """解析坐标字符串.

    格式: "x1,y1 x2,y2 x3,y3 x4,y4" 例如: "100,50 1800,80 1850,1000 120,1050"
    """
    points = []
    try:
        for point_str in coord_str.split():
            x, y = point_str.split(",")
            points.append((float(x), float(y)))
    except Exception as e:
        raise ValueError(f"坐标格式错误: {e}. 正确格式应为: 'x1,y1 x2,y2 x3,y3 x4,y4'")

    if len(points) != 4:
        raise ValueError(f"需要4个坐标点，实际提供了 {len(points)} 个")

    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homography标定工具 - 输入坐标计算变换矩阵")
    parser.add_argument("--pixel-points", type=str, required=True, help='像素坐标 (格式: "x1,y1 x2,y2 x3,y3 x4,y4")')
    parser.add_argument(
        "--world-points", type=str, required=True, help='世界坐标 (格式: "x1,y1 x2,y2 x3,y3 x4,y4", 单位:米)'
    )
    parser.add_argument("--video-name", type=str, default="calibration", help="视频名称（用于输出文件名）")
    parser.add_argument("--output", type=str, default="calibration", help="输出目录")

    args = parser.parse_args()

    try:
        # 解析坐标
        pixel_points = parse_coordinates(args.pixel_points)
        world_points = parse_coordinates(args.world_points)

        print(f"像素坐标: {pixel_points}")
        print(f"世界坐标: {world_points}")

        # 执行标定
        calibrator = HomographyCalibrator(pixel_points, world_points, args.output)
        H = calibrator.compute_homography()

        if H is not None:
            calibrator.save_homography(H, args.video_name)
            calibrator.test_transform(H)
            print("\n✓ 标定完成！")
        else:
            print("❌ 标定失败")
            exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        exit(1)
