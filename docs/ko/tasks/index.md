---
comments: true
description: YOLOv8을 사용하여 수행할 수 있는 컴퓨터 비전 작업의 기초인 탐지, 세분화, 분류 및 자세 추정에 대해 알아보세요. AI 프로젝트에서의 그 용도를 이해하세요.
keywords: Ultralytics, YOLOv8, 탐지, 세분화, 분류, 자세 추정, AI 프레임워크, 컴퓨터 비전 작업
---

# Ultralytics YOLOv8 작업

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO 지원 작업">

YOLOv8는 여러 컴퓨터 비전 **작업**을 지원하는 AI 프레임워크입니다. 이 프레임워크는 [탐지](detect.md), [세분화](segment.md), [분류](classify.md), 그리고 [자세](pose.md) 추정을 수행하는 데 사용될 수 있습니다. 각각의 작업은 서로 다른 목적과 사용 사례를 가지고 있습니다.

!!! Note "노트"

    🚧 다국어 문서화 작업이 진행 중에 있으며, 더 나은 문서를 제공하기 위해 노력하고 있습니다. 인내해 주셔서 감사합니다! 🙏

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>보기:</strong> Ultralytics YOLO 작업 탐색: 객체 탐지, 세분화, 추적, 자세 추정.
</p>

## [탐지](detect.md)

탐지는 YOLOv8이 지원하는 기본 작업입니다. 이미지 또는 비디오 프레임에서 객체를 탐지하고 주변에 경계 상자를 그리는 것을 포함합니다. 탐지된 객체들은 그 특징에 따라 다른 카테고리로 분류됩니다. YOLOv8은 단일 이미지나 비디오 프레임에서 여러 객체를 정확하고 빠르게 탐지할 수 있습니다.

[탐지 예시](detect.md){ .md-button }

## [세분화](segment.md)

세분화는 이미지를 내용에 기반하여 다른 영역으로 나누는 작업입니다. 각 영역은 내용에 따라 레이블이 지정됩니다. 이 작업은 이미지 세분화와 의료 영상과 같은 응용 분야에 유용합니다. YOLOv8는 U-Net 아키텍처의 변형을 사용하여 세분화를 수행합니다.

[세분화 예시](segment.md){ .md-button }

## [분류](classify.md)

분류는 이미지를 다른 카테고리로 분류하는 작업입니다. YOLOv8는 이미지의 내용을 바탕으로 이미지 분류에 사용될 수 있습니다. 이는 EfficientNet 아키텍처의 변형을 사용하여 분류 작업을 수행합니다.

[분류 예시](classify.md){ .md-button }

## [자세](pose.md)

자세/키포인트 탐지는 이미지나 비디오 프레임에서 특정 점들을 탐지하는 작업입니다. 이들 점은 키포인트로 불리며, 움직임 추적이나 자세 추정에 사용됩니다. YOLOv8은 이미지나 비디오 프레임의 키포인트를 정확하고 빠르게 탐지할 수 있습니다.

[자세 예시](pose.md){ .md-button }

## 결론

YOLOv8은 탐지, 세분화, 분류, 키포인트 탐지 등 다양한 작업을 지원합니다. 각각의 작업은 다른 목적과 사용 사례를 가지고 있습니다. 이러한 작업의 차이점을 이해함으로써, 컴퓨터 비전 응용 프로그램에 적합한 작업을 선택할 수 있습니다.
