# Warped Videos Output

这个文件夹存储使用 homography_transform_video.py 生成的鸟瞰视图视频。

## 文件说明

### Homograph_Teset_FullScreen_warped_new.mp4
- **分辨率**: 180×1200 (鸟瞰视图)
- **帧数**: 154
- **FPS**: 30
- **文件大小**: 401.3 KB
- **生成日期**: 2026-01-04
- **验证状态**: ✓ 完整有效

## 生成方法

```bash
python homography_transform_video.py \
  --input videos/Homograph_Teset_FullScreen.mp4 \
  --homography calibration/Homograph_Teset_FullScreen_homography.json \
  --output results/warped_videos/Homograph_Teset_FullScreen_warped_new.mp4
```

## 技术细节

- 使用手工逐像素变换 + 双线性插值
- Homography矩阵已验证：所有4个参考点误差 < 0.000001m
- 输出坐标系映射：180×1200 → 7.5m × 50m (世界坐标)
