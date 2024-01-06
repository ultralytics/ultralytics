---
comments: true
description: Ultralytics YOLO를 사용하여 비디오 스트림에서 객체 추적을 사용하는 방법을 알아보세요. 다양한 추적기를 사용하는 안내와 추적기 구성을 맞춤 설정하는 방법에 대한 가이드가 있습니다.
keywords: Ultralytics, YOLO, 객체 추적, 비디오 스트림, BoT-SORT, ByteTrack, 파이썬 가이드, CLI 가이드
---

# Ultralytics YOLO를 이용한 다중 객체 추적

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="다중 객체 추적 예시">

비디오 분석의 영역에서 객체 추적은 프레임 내에서 객체의 위치와 클래스를 식별할 뿐만 아니라 비디오가 진행됨에 따라 각각의 검출된 객체에 대해 고유 ID를 유지하는 중요한 작업입니다. 응용 프로그램은 감시 및 보안에서 실시간 스포츠 분석에 이르기까지 무한합니다.

## 객체 추적을 위해 Ultralytics YOLO를 선택해야 하는 이유는?

Ultralytics 추적기의 출력은 표준 객체 검출과 일관되지만 객체 ID가 추가된 가치가 있습니다. 이를 통해 비디오 스트림에서 객체를 추적하고 이후 분석을 수행하기가 쉽습니다. 여기에 어떤 이유로 Ultralytics YOLO를 사용해야 하는지에 대해 설명합니다:

- **효율성:** 정확성을 저하시키지 않으면서 실시간으로 비디오 스트림을 처리합니다.
- **유연성:** 다양한 추적 알고리즘과 구성을 지원합니다.
- **사용하기 쉬움:** 간단한 파이썬 API 및 CLI 옵션으로 빠른 통합 및 배치가 가능합니다.
- **맞춤 설정:** 맞춤 학습된 YOLO 모델과 함께 사용하기 쉬워 특정 도메인 응용 프로그램에 통합할 수 있습니다.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube 비디오 플레이어" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>시청하기:</strong> Ultralytics YOLOv8로 객체 감지 및 추적하기.
</p>

## 실제 세계 응용 프로그램

|                                                    교통수단                                                     |                                                     소매업                                                     |                                                     수산업                                                      |
|:-----------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|
| ![차량 추적](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![사람 추적](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![물고기 추적](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    차량 추적                                                    |                                                    사람 추적                                                    |                                                    물고기 추적                                                    |

## 한눈에 보기

Ultralytics YOLO는 객체 감지 기능을 확장하여 견고하고 다재다능한 객체 추적을 제공합니다:

- **실시간 추적:** 고화면률의 비디오에서 매끄럽게 객체 추적합니다.
- **다중 추적기 지원:** 다양한 검증된 추적 알고리즘 중에서 선택 가능합니다.
- **맞춤형 추적기 구성:** 다양한 매개변수를 조정하여 특정 요구사항에 맞게 추적 알고리즘을 맞춤화할 수 있습니다.

## 사용 가능한 추적기

Ultralytics YOLO는 다음과 같은 추적 알고리즘을 지원합니다. 관련 YAML 구성 파일(예: `tracker=tracker_type.yaml`)을 전달하여 사용할 수 있습니다:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 이 추적기를 활성화하려면 `botsort.yaml`을 사용합니다.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - 이 추적기를 활성화하려면 `bytetrack.yaml`을 사용합니다.

기본 추적기는 BoT-SORT입니다.

## 추적

비디오 스트림에서 추적기를 실행하려면 YOLOv8n, YOLOv8n-seg 및 YOLOv8n-pose와 같은 훈련된 Detect, Segment 또는 Pose 모델을 사용하십시오.

!!! 예시 ""

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 공식 모델 또는 맞춤 모델을 불러오기
        model = YOLO('yolov8n.pt')  # 공식 Detect 모델 불러오기
        model = YOLO('yolov8n-seg.pt')  # 공식 Segment 모델 불러오기
        model = YOLO('yolov8n-pose.pt')  # 공식 Pose 모델 불러오기
        model = YOLO('path/to/best.pt')  # 맞춤 학습된 모델 불러오기

        # 모델을 사용하여 추적 수행
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # 기본 추적기로 추적하기
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # ByteTrack 추적기로 추적하기
        ```

    === "CLI"

        ```bash
        # 명령 행 인터페이스를 사용하여 다양한 모델로 추적 수행
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # 공식 Detect 모델
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 공식 Segment 모델
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # 공식 Pose 모델
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # 맞춤 학습된 모델

        # ByteTrack 추적기를 사용하여 추적하기
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

위의 사용법에서 볼 수 있듯이 모든 Detect, Segment 및 Pose 모델은 비디오나 스트리밍 출처에서 추적이 가능합니다.

## 구성

### 추적 인수

추적 구성은 `conf`, `iou` 및 `show`와 같은 예측 모드와 동일한 속성을 공유합니다. 추가 구성에 대해서는 [Predict](https://docs.ultralytics.com/modes/predict/) 모델 페이지를 참조하십시오.

!!! 예시 ""

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 추적 매개변수를 구성하고 추적기를 실행합니다
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # 추적 매개변수를 구성하고 명령 행 인터페이스를 사용하여 추적기를 실행합니다
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### 추적기 선택

Ultralytics에서는 수정된 추적기 구성 파일도 사용할 수 있습니다. 이를 위해 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)에서 추적기 구성 파일의 복사본(예: `custom_tracker.yaml`)을 만들고 필요한대로 구성을 수정하면 됩니다(단, `tracker_type` 제외).

!!! 예시 ""

    === "파이썬"

        ```python
        from ultralytics import YOLO

        # 모델을 불러오고 맞춤 구성 파일로 추적기를 실행합니다
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # 명령 행 인터페이스를 사용하여 맞춤 구성 파일로 모델을 불러오고 추적기를 실행합니다
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

추적 인수에 대한 종합적인 목록은 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 페이지를 참조하세요.

## 파이썬 예시

### 보존하는 추적 루프

다음은 OpenCV(`cv2`)와 YOLOv8를 사용하여 비디오 프레임에서 객체 추적을 실행하는 파이썬 스크립트입니다. 이 스크립트에서는 필요한 패키지(`opencv-python` 및 `ultralytics`)를 이미 설치했다고 가정합니다. `persist=True` 인수는 추적기에 현재 이미지 또는 프레임이 시퀀스에서 다음 것이며 현재 이미지에서 이전 이미지의 추적을 예상한다고 알립니다.

!!! 예시 "추적이 포함된 스트리밍 for-loop"

    ```python
    import cv2
    from ultralytics import YOLO

    # YOLOv8 모델을 불러옵니다
    model = YOLO('yolov8n.pt')

    # 비디오 파일을 엽니다
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 비디오 프레임을 반복합니다
    while cap.isOpened():
        # 비디오에서 프레임을 읽습니다
        success, frame = cap.read()

        if success:
            # 프레임에 YOLOv8 추적을 실행하여 추적을 유지합니다
            results = model.track(frame, persist=True)

            # 결과를 프레임에 시각화합니다
            annotated_frame = results[0].plot()

            # 어노테이션된 프레임을 표시합니다
            cv2.imshow("YOLOv8 추적", annotated_frame)

            # 'q'가 눌리면 루프를 중단합니다
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 비디오의 끝에 도달하면 루프를 중단합니다
            break

    # 비디오 캡처 객체를 해제하고 표시 창을 닫습니다
    cap.release()
    cv2.destroyAllWindows()
    ```

여기서 `model(frame)`을 `model.track(frame)`으로 변경하면 단순 감지가 아닌 객체 추적이 가능해집니다. 이 수정된 스크립트는 비디오의 각 프레임에 추적기를 실행하고 결과를 시각화한 후 창에 표시합니다. 'q'를 누르면 루프가 종료됩니다.

### 시간에 따른 추적 그리기

연속 프레임에서 객체 추적을 시각화하면 비디오 내에서 검출된 객체의 이동 패턴과 행동에 대한 소중한 통찰력을 제공할 수 있습니다. Ultralytics YOLOv8을 사용하면 이러한 추적을 원활하고 효율적으로 플로팅할 수 있습니다.

다음 예시에서, 여러 비디오 프레임에 걸친 검출된 객체의 움직임을 플로팅하기 위해 YOLOv8의 추적 기능을 활용하는 방법을 보여줍니다. 이 스크립트는 비디오 파일을 여는 것을 포함하여 프레임별로 읽고 YOLO 모델을 사용하여 다양한 객체를 식별하고 추적합니다. 검출된 경계 상자의 중심점을 보존하고 연결하여 추적된 객체의 경로를 나타내는 선을 그립니다.

!!! 예시 "비디오 프레임에 걸쳐 추적 그리기"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # YOLOv8 모델을 불러옵니다
    model = YOLO('yolov8n.pt')

    # 비디오 파일을 엽니다
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 추적 내역을 저장합니다
    track_history = defaultdict(lambda: [])

    # 비디오 프레임을 반복합니다
    while cap.isOpened():
        # 비디오에서 프레임을 읽습니다
        success, frame = cap.read()

        if success:
            # 프레임에 YOLOv8 추적을 실행하여 추적을 유지합니다
            results = model.track(frame, persist=True)

            # 상자 및 추적 ID를 가져옵니다
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 결과를 프레임에 시각화합니다
            annotated_frame = results[0].plot()

            # 추적을 플롯합니다
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y의 중심점
                if len(track) > 30:  # 90프레임에 대해 90개의 추적을 유지
                    track.pop(0)

                # 추적 라인을 그립니다
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 어노테이션된 프레임을 표시합니다
            cv2.imshow("YOLOv8 추적", annotated_frame)

            # 'q'가 눌리면 루프를 중단합니다
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 비디오의 끝에 도달하면 루프를 중단합니다
            break

    ```
