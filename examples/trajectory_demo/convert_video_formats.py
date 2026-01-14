"""
convert_video_formats.py.

将AVI视频转换为多种格式
"""

from pathlib import Path

import cv2


def convert_video(input_path, output_path):
    """转换视频格式."""
    print(f"读取: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ 无法打开输入视频")
        return False

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"输入信息: {width}×{height}, {total_frames}帧, {fps:.2f}FPS")

    # 获取扩展名并选择编码
    ext = Path(output_path).suffix.lower()

    if ext == ".mp4":
        # MP4 - 尝试多种编码
        codecs = [
            cv2.VideoWriter_fourcc(*"mp4v"),
            cv2.VideoWriter_fourcc(*"avc1"),
            cv2.VideoWriter_fourcc(*"H264"),
        ]
    elif ext == ".avi":
        # AVI
        codecs = [
            cv2.VideoWriter_fourcc(*"MJPG"),  # Motion JPEG
            cv2.VideoWriter_fourcc(*"DIVX"),
            cv2.VideoWriter_fourcc(*"XVID"),
        ]
    elif ext == ".mov":
        # MOV
        codecs = [
            cv2.VideoWriter_fourcc(*"mp4v"),
            cv2.VideoWriter_fourcc(*"avc1"),
        ]
    else:
        print(f"❌ 不支持的格式: {ext}")
        return False

    # 尝试编码
    out = None
    for codec in codecs:
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        if out.isOpened():
            codec_name = (
                chr(codec & 0xFF) + chr((codec >> 8) & 0xFF) + chr((codec >> 16) & 0xFF) + chr((codec >> 24) & 0xFF)
            )
            print(f"✓ 使用编码: {codec_name.strip()}")
            break

    if out is None or not out.isOpened():
        print("❌ 无法创建输出视频")
        cap.release()
        return False

    # 逐帧复制
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  进度: {frame_count}/{total_frames}")

    cap.release()
    out.release()

    print(f"✓ 转换完成: {output_path}")
    print(f"  处理了 {frame_count} 帧")

    return True


if __name__ == "__main__":
    input_file = "/workspace/ultralytics/videos/warped_test_360x2400.avi"

    # 转换为多种格式
    outputs = [
        "/workspace/ultralytics/videos/warped_test_360x2400.mov",  # MOV格式
        "/workspace/ultralytics/videos/warped_test_360x2400_mjpg.avi",  # MJPG编码AVI
    ]

    print("=" * 70)
    print("视频格式转换")
    print("=" * 70)

    for output in outputs:
        print(f"\n转换到: {output}")
        convert_video(input_file, output)

    print("\n" + "=" * 70)
    print("✓ 转换完成")
    print("=" * 70)
