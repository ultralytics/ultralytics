---
comments: true
description: YOLOv8 예측 모드를 사용하여 다양한 작업을 수행하는 방법을 알아보십시오. 이미지, 비디오 및 데이터 형식과 같은 다양한 추론 소스에 대해 자세히 알아봅니다.
keywords: Ultralytics, YOLOv8, 예측 모드, 추론 소스, 예측 작업, 스트리밍 모드, 이미지 처리, 비디오 처리, 머신 러닝, AI
---

# Ultralytics YOLO로 모델 예측

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 생태계와 통합">

## 소개

머신 러닝 및 컴퓨터 비전의 세계에서 시각적 데이터를 해석하는 과정을 '추론' 또는 '예측'이라고 합니다. Ultralytics YOLOv8는 다양한 데이터 소스에서의 고성능, 실시간 추론을 위해 맞춤화된 강력한 기능인 **예측 모드**를 제공합니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청:</strong> Ultralytics YOLOv8 모델에서 출력을 추출하여 맞춤 프로젝트에 사용하는 방법.
</p>

## 실제 응용 분야

|                                                        제조업                                                        |                                                      스포츠                                                       |                                                       안전                                                        |
|:-----------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
| ![차량 예비 부품 탐지](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![축구 선수 탐지](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![사람 넘어짐 탐지](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                    차량 예비 부품 탐지                                                    |                                                    축구 선수 탐지                                                    |                                                    사람 넘어짐 탐지                                                    |

## 예측 인퍼런스를 위해 Ultralytics YOLO 사용하기

다음은 YOLOv8의 예측 모드를 다양한 추론 요구 사항에 사용해야 하는 이유입니다:

- **다양성:** 이미지, 비디오, 심지어 실시간 스트림에 대한 추론을 수행할 수 있습니다.
- **성능:** 정확성을 희생하지 않고 실시간, 고속 처리를 위해 설계되었습니다.
- **사용 편의성:** 빠른 배포 및 테스트를 위한 직관적인 Python 및 CLI 인터페이스를 제공합니다.
- **고도의 사용자 정의:** 특정 요구 사항에 맞게 모델의 추론 행동을 조율하기 위한 다양한 설정 및 매개변수를 제공합니다.

### 예측 모드의 주요 기능

YOLOv8의 예측 모드는 강력하고 다재다능하게 설계되었으며, 다음과 같은 특징을 갖고 있습니다:

- **다중 데이터 소스 호환성:** 데이터가 개별 이미지, 이미지 컬렉션, 비디오 파일 또는 실시간 비디오 스트림의 형태로 존재하는지 여부에 관계없이 예측 모드가 지원합니다.
- **스트리밍 모드:** `Results` 객체의 메모리 효율적인 생성자로 스트리밍 기능을 사용합니다. 예측기의 호출 메서드에서 `stream=True`로 설정하여 활성화합니다.
- **배치 처리:** 단일 배치에서 여러 이미지 또는 비디오 프레임을 처리하는 기능을 통해 추론 시간을 더욱 단축합니다.
- **통합 친화적:** 유연한 API 덕분에 기존 데이터 파이프라인 및 기타 소프트웨어 구성 요소와 쉽게 통합할 수 있습니다.

Ultralytics YOLO 모델은 Python `Results` 객체의 리스트를 반환하거나, 추론 중 `stream=True`가 모델에 전달될 때 `Results` 객체의 메모리 효율적인 Python 생성자를 반환합니다:

!!! 예시 "예측"

    === "`stream=False`로 리스트 반환"
        ```python
        from ultralytics import YOLO

        # 모델 로드
        model = YOLO('yolov8n.pt')  # 사전 훈련된 YOLOv8n 모델

        # 이미지 리스트에 대한 배치 추론 실행
        results = model(['im1.jpg', 'im2.jpg'])  # Results 객체의 리스트 반환

        # 결과 리스트 처리
        for result in results:
            boxes = result.boxes  # bbox 출력을 위한 Boxes 객체
            masks = result.masks  # 세그멘테이션 마스크 출력을 위한 Masks 객체
            keypoints = result.keypoints  # 자세 출력을 위한 Keypoints 객체
            probs = result.probs  # 분류 출력을 위한 Probs 객체
        ```

    === "`stream=True`로 생성자 반환"
        ```python
        from ultralytics import YOLO

        # 모델 로드
        model = YOLO('yolov8n.pt')  # 사전 훈련된 YOLOv8n 모델

        # 이미지 리스트에 대한 배치 추론 실행
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # Results 객체의 생성자 반환

        # 결과 생성자 처리
        for result in results:
            boxes = result.boxes  # bbox 출력을 위한 Boxes 객체
            masks = result.masks  # 세그멘테이션 마스크 출력을 위한 Masks 객체
            keypoints = result.keypoints  # 자세 출력을 위한 Keypoints 객체
            probs = result.probs  # 분류 출력을 위한 Probs 객체
        ```

## 추론 소스

YOLOv8은 아래 표에 표시된 바와 같이 추론을 위한 다양한 유형의 입력 소스를 처리할 수 있습니다. 소스에는 정적 이미지, 비디오 스트림, 다양한 데이터 형식이 포함됩니다. 표는 또한 각 소스를 'stream=True' ✅와 함께 스트리밍 모드에서 사용할 수 있는지 여부를 나타냅니다. 스트리밍 모드는 비디오나 라이브 스트림을 처리할 때 결과를 메모리에 모두 로드하는 대신 결과의 생성자를 만들어 유용하게 사용됩니다.

!!! Tip "팁"

    긴 비디오나 큰 데이터 세트를 처리할 때 'stream=True'를 사용하여 효율적으로 메모리를 관리합니다. 'stream=False'일 때는 모든 프레임 또는 데이터 포인트에 대한 결과가 메모리에 저장되어, 입력이 크면 메모리 부족 오류를 빠르게 유발할 수 있습니다. 반면에, 'stream=True'는 생성자를 사용하여 현재 프레임 또는 데이터 포인트의 결과만 메모리에 유지하여 메모리 소비를 크게 줄이고 메모리 부족 문제를 방지합니다.

| 소스        | 인수                                         | 유형              | 비고                                                                       |
|-----------|--------------------------------------------|-----------------|--------------------------------------------------------------------------|
| 이미지       | `'image.jpg'`                              | `str` 또는 `Path` | 단일 이미지 파일.                                                               |
| URL       | `'https://ultralytics.com/images/bus.jpg'` | `str`           | 이미지 URL.                                                                 |
| 스크린샷      | `'screen'`                                 | `str`           | 스크린샷을 캡처합니다.                                                             |
| PIL       | `Image.open('im.jpg')`                     | `PIL.Image`     | HWC 형식으로 RGB 채널이 있습니다.                                                   |
| OpenCV    | `cv2.imread('im.jpg')`                     | `np.ndarray`    | HWC 형식으로 BGR 채널이 있고 `uint8 (0-255)` 입니다.                                 |
| numpy     | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC 형식으로 BGR 채널이 있고 `uint8 (0-255)` 입니다.                                 |
| torch     | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW 형식으로 RGB 채널이 있고 `float32 (0.0-1.0)` 입니다.                            |
| CSV       | `'sources.csv'`                            | `str` 또는 `Path` | 이미지, 비디오 또는 디렉토리 경로가 있는 CSV 파일.                                          |
| 비디오 ✅     | `'video.mp4'`                              | `str` 또는 `Path` | MP4, AVI 등과 같은 형식의 비디오 파일입니다.                                            |
| 디렉토리 ✅    | `'path/'`                                  | `str` 또는 `Path` | 이미지나 비디오가 있는 디렉토리 경로입니다.                                                 |
| 글로브 ✅     | `'path/*.jpg'`                             | `str`           | 여러 파일에 일치하는 글로브 패턴입니다. '*' 문자를 와일드카드로 사용하세요.                             |
| YouTube ✅ | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | YouTube 비디오의 URL입니다.                                                     |
| 스트림 ✅     | `'rtsp://example.com/media.mp4'`           | `str`           | RTSP, RTMP, TCP 또는 IP 주소와 같은 스트리밍 프로토콜의 URL입니다.                          |
| 멀티-스트림 ✅  | `'list.streams'`                           | `str` 또는 `Path` | 스트림 URL이 행당 하나씩 있는 `*.streams` 텍스트 파일이며, 예를 들어 8개의 스트림은 배치 크기 8에서 실행됩니다. |

아래는 각 유형의 소스를 사용하는 코드 예제입니다:

!!! 예시 "예측 소스"

    === "이미지"
        이미지 파일에서 추론을 실행합니다.
        ```python
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 이미지 파일 경로 정의
        source = 'path/to/image.jpg'

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "스크린샷"
        현재 스크린 콘텐츠를 스크린샷으로 추론을 실행합니다.
        ```python
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 현재 스크린샷을 소스로 정의
        source = 'screen'

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "URL"
        URL을 통해 원격으로 호스팅되는 이미지나 비디오에서 추론을 실행합니다.
        ```python
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 원격 이미지나 동영상 URL 정의
        source = 'https://ultralytics.com/images/bus.jpg'

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "PIL"
        Python Imaging Library (PIL)로 열린 이미지에서 추론을 실행합니다.
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # PIL을 사용하여 이미지 열기
        source = Image.open('path/to/image.jpg')

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "OpenCV"
        OpenCV로 읽은 이미지에서 추론을 실행합니다.
        ```python
        import cv2
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # OpenCV를 사용하여 이미지 읽기
        source = cv2.imread('path/to/image.jpg')

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "numpy"
        numpy 배열로 표현된 이미지에서 추론을 실행합니다.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 무작위 numpy 배열 생성, HWC 형태 (640, 640, 3), 값 범위 [0, 255], 타입 uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```

    === "torch"
        PyTorch 텐서로 표현된 이미지에서 추론을 실행합니다.
        ```python
        import torch
        from ultralytics import YOLO

        # 사전 훈련된 YOLOv8n 모델 로드
        model = YOLO('yolov8n.pt')

        # 무작위 torch 텐서 생성, BCHW 형태 (1, 3, 640, 640), 값 범위 [0, 1], 타입 float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # 소스에서 추론 실행
        results = model(source)  # Results 객체의 리스트
        ```
