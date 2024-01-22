---
comments: true
description: Ultralytics가 지원하는 객체 탐지, 세분화, 자세 추정, 이미지 분류, 다중 객체 추적을 위한 다양한 컴퓨터 비전 데이터셋에 대한 탐색입니다.
keywords: 컴퓨터 비전, 데이터셋, Ultralytics, YOLO, 객체 탐지, 인스턴스 세분화, 자세 추정, 이미지 분류, 다중 객체 추적
---

# 데이터셋 개요

Ultralytics는 탐지, 인스턴스 세분화, 자세 추정, 분류 및 다중 객체 추적과 같은 컴퓨터 비전 작업을 촉진하기 위해 다양한 데이터셋을 지원합니다. 아래는 주요 Ultralytics 데이터셋 목록과 각 컴퓨터 비전 작업의 요약, 그리고 해당 데이터셋입니다.

!!! Note "노트"

    🚧 다국어 문서 작업은 현재 진행 중이며, 우리는 이를 개선하기 위해 노력하고 있습니다. 인내해 주셔서 감사합니다! 🙏

## [탐지 데이터셋](../../datasets/detect/index.md)

바운딩 박스 객체 탐지는 이미지 내 객체들을 탐지하고 각 객체 주위에 바운딩 박스를 그려 객체를 위치시키는 컴퓨터 비전 기법입니다.

- [Argoverse](../../datasets/detect/argoverse.md): 도시 환경에서의 3D 추적 및 동작 예측 데이터와 풍부한 주석이 담긴 데이터셋입니다.
- [COCO](../../datasets/detect/coco.md): 20만개가 넘는 레이블이 붙은 이미지로 설계된 객체 탐지, 세분화 및 설명을 위한 대규모 데이터셋입니다.
- [COCO8](../../datasets/detect/coco8.md): COCO train 및 COCO val에서 처음 4개의 이미지를 포함하여 신속한 테스트에 적합합니다.
- [Global Wheat 2020](../../datasets/detect/globalwheat2020.md): 전 세계에서 수집한 밀 머리 이미지로 구성된 객체 탐지 및 위치 지정 작업을 위한 데이터셋입니다.
- [Objects365](../../datasets/detect/objects365.md): 365개 객체 카테고리와 60만개가 넘는 주석이 달린 이미지를 포함하는 고품질 대규모 객체 탐지용 데이터셋입니다.
- [OpenImagesV7](../../datasets/detect/open-images-v7.md): 구글에서 제공하는 170만개의 훈련 이미지와 4만 2천개의 검증 이미지를 포함하는 포괄적인 데이터셋입니다.
- [SKU-110K](../../datasets/detect/sku-110k.md): 1만 1천 개의 이미지와 170만 개의 바운딩 박스를 특징으로 하는 소매 환경에서 밀집된 객체 탐지를 위한 데이터셋입니다.
- [VisDrone](../../datasets/detect/visdrone.md): 드론으로 촬영한 영상에서 객체 탐지 및 다중 객체 추적 데이터를 포함하는 데이터셋으로 1만 개가 넘는 이미지와 비디오 시퀀스를 포함합니다.
- [VOC](../../datasets/detect/voc.md): 객체 탐지와 세분화를 위한 파스칼 시각 객체 클래스(VOC) 데이터셋으로 20개 객체 클래스와 1만 1천 개가 넘는 이미지를 포함합니다.
- [xView](../../datasets/detect/xview.md): 상공 이미지에서 객체 탐지를 위한 데이터셋으로 60개 객체 카테고리와 100만 개가 넘는 주석이 달린 객체를 포함합니다.

## [인스턴스 세분화 데이터셋](../../datasets/segment/index.md)

인스턴스 세분화는 이미지 내 객체들을 픽셀 수준에서 식별하고 위치를 지정하는 컴퓨터 비전 기법입니다.

- [COCO](../../datasets/segment/coco.md): 객체 탐지, 세분화 및 설명 작업을 위해 설계된 20만개가 넘는 레이블이 붙은 이미지로 구성된 대규모 데이터셋입니다.
- [COCO8-seg](../../datasets/segment/coco8-seg.md): 세분화 주석이 있는 8개의 COCO 이미지로 구성된 인스턴스 세분화 작업을 위한 더 작은 데이터셋입니다.

## [자세 추정](../../datasets/pose/index.md)

자세 추정은 카메라 또는 세계 좌표계에 대한 객체의 자세를 결정하는 기술입니다.

- [COCO](../../datasets/pose/coco.md): 자세 추정 작업을 위해 사람의 자세 주석이 달린 대규모 데이터셋입니다.
- [COCO8-pose](../../datasets/pose/coco8-pose.md): 인간의 자세 주석이 포함된 8개의 COCO 이미지로 구성된 자세 추정 작업을 위한 더 작은 데이터셋입니다.
- [Tiger-pose](../../datasets/pose/tiger-pose.md): 자세 추정 작업을 위한 호랑이를 포함한 263개 이미지로 구성된 컴팩트한 데이터셋으로, 호랑이당 12개의 키포인트가 주석으로 표시되어 있습니다.

## [분류](../../datasets/classify/index.md)

이미지 분류는 이미지의 시각적 내용을 기반으로 이미지를 하나 이상의 미리 정의된 클래스나 카테고리로 분류하는 컴퓨터 비전 작업입니다.

- [Caltech 101](../../datasets/classify/caltech101.md): 이미지 분류 작업을 위한 101개의 객체 카테고리를 포함하는 데이터셋입니다.
- [Caltech 256](../../datasets/classify/caltech256.md): Caltech 101의 확장판으로 256개의 객체 카테고리와 더 어려운 이미지를 포함합니다.
- [CIFAR-10](../../datasets/classify/cifar10.md): 각 클래스당 6천 개의 이미지를 포함하는 10개의 클래스로 구성된 60K 32x32 컬러 이미지 데이터셋입니다.
- [CIFAR-100](../../datasets/classify/cifar100.md): CIFAR-10의 확장판으로 100개의 객체 카테고리 및 클래스 당 600개 이미지를 포함합니다.
- [Fashion-MNIST](../../datasets/classify/fashion-mnist.md): 이미지 분류 작업을 위한 10개 패션 카테고리의 7만 개 그레이스케일 이미지를 포함하는 데이터셋입니다.
- [ImageNet](../../datasets/classify/imagenet.md): 객체 탐지 및 이미지 분류를 위한 1400만 개의 이미지와 2만 개의 카테고리를 포함하는 대규모 데이터셋입니다.
- [ImageNet-10](../../datasets/classify/imagenet10.md): 실험 및 테스트 속도를 높이기 위한 ImageNet의 10개 카테고리를 포함하는 더 작은 하위 집합입니다.
- [Imagenette](../../datasets/classify/imagenette.md): 훈련과 테스트를 더 빠르게 진행할 수 있도록 쉽게 구별 가능한 클래스 10개를 포함하는 ImageNet의 더 작은 하위 집합입니다.
- [Imagewoof](../../datasets/classify/imagewoof.md): 이미지 분류 작업을 위한 ImageNet의 더 어려운 하위 집합으로, 10개의 개 품종 카테고리를 포함합니다.
- [MNIST](../../datasets/classify/mnist.md): 손으로 쓴 숫자의 7만 개 그레이스케일 이미지를 포함하는 이미지 분류 작업을 위한 데이터셋입니다.

## [지향 바운딩 박스 (OBB)](../../datasets/obb/index.md)

지향 바운딩 박스(OBB)는 이미지 내 비스듬한 객체를 회전된 바운딩 박스를 사용하여 탐지하는 컴퓨터 비전 방법으로, 종종 항공 및 위성 영상에 적용됩니다.

- [DOTAv2](../../datasets/obb/dota-v2.md): 170만 개 인스턴스와 1만 1천 268개 이미지를 포함하는 인기 있는 OBB 항공 이미지 데이터셋입니다.

## [다중 객체 추적](../../datasets/track/index.md)

다중 객체 추적은 비디오 시퀀스에서 시간에 걸쳐 여러 객체를 탐지하고 추적하는 컴퓨터 비전 기술입니다.

- [Argoverse](../../datasets/detect/argoverse.md): 도시 환경에서의 3D 추적 및 동작 예측 데이터와 풍부한 주석으로 다중 객체 추적 작업을 위한 데이터셋입니다.
- [VisDrone](../../datasets/detect/visdrone.md): 드론으로 촬영한 영상에서 객체 탐지 및 다중 객체 추적 데이터를 포함하는 데이터셋으로 1만 개가 넘는 이미지와 비디오 시퀀스를 포함합니다.

## 새 데이터셋 기여하기

새 데이터셋을 기여하는 것은 기존 인프라와 잘 조화되도록 보장하기 위해 여러 단계를 포함합니다. 아래는 필요한 단계입니다:

### 새 데이터셋 기여를 위한 단계

1. **이미지 수집**: 데이터셋에 속하는 이미지를 모읍니다. 이는 공공 데이터베이스 또는 자체 수집한 자료 등 다양한 출처에서 수집할 수 있습니다.

2. **이미지 주석 달기**: 이러한 이미지에 작업에 따라 바운딩 박스, 세그먼트 또는 키포인트로 주석을 답니다.

3. **주석 내보내기**: 이 주석들을 Ultralytics가 지원하는 YOLO `*.txt` 파일 형식으로 변환합니다.

4. **데이터셋 구성**: 데이터셋을 올바른 폴더 구조로 배열합니다. 'train/'과 'val/' 상위 디렉토리를 갖고 있어야 하며, 각각 'images/' 및 'labels/' 하위 디렉토리가 있어야 합니다.

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **`data.yaml` 파일 생성**: 데이터셋의 루트 디렉토리에서 데이터셋, 클래스 및 기타 필요한 정보를 설명하는 `data.yaml` 파일을 만듭니다.

6. **이미지 최적화 (선택)**: 처리 과정을 더 효율적으로 하기 위해 데이터셋의 크기를 줄이고자 한다면 아래의 코드를 사용하여 이미지를 최적화할 수 있습니다. 필수는 아니지만, 데이터셋 크기를 작게하고 다운로드 속도를 빠르게 하는 것이 추천됩니다.

7. **데이터셋 압축**: 전체 데이터셋 폴더를 zip 파일로 압축합니다.

8. **문서화 및 PR**: 데이터셋에 대한 설명과 기존 프레임워크와의 적합성에 대해 설명하는 문서화 페이지를 만든 다음, 풀 리퀘스트(PR)를 제출합니다. PR을 제출하는 더 자세한 방법에 관해서는 [Ultralytics 기여 가이드라인](https://docs.ultralytics.com/help/contributing)을 참고하십시오.

### 데이터셋 최적화 및 압축 예제 코드

!!! Example "데이터셋 최적화 및 압축"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # 데이터셋 디렉토리 정의
    path = Path('path/to/dataset')

    # 데이터셋의 이미지 최적화 (선택사항)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # 'path/to/dataset.zip'으로 데이터셋 압축
    zip_directory(path)
    ```

이 단계들을 따르면 Ultralytics의 기존 구조와 잘 통합되는 새로운 데이터셋을 기여할 수 있습니다.
