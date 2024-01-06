---
comments: true
description: YOLOv8が実行できる基本的なコンピュータービジョンタスクについて学び、検出、セグメンテーション、分類、ポーズ認識がAIプロジェクトでどのように使用されるかを理解します。
keywords: Ultralytics, YOLOv8, 検出, セグメンテーション, 分類, ポーズ推定, AIフレームワーク, コンピュータービジョンタスク
---

# Ultralytics YOLOv8タスク

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLOがサポートするタスク">

YOLOv8は、複数のコンピュータービジョン**タスク**をサポートするAIフレームワークです。このフレームワークは、[検出](detect.md)、[セグメンテーション](segment.md)、[分類](classify.md)、及び[ポーズ](pose.md)推定を実行するために使用できます。これらのタスクはそれぞれ異なる目的と用途を持っています。

!!! Note "ノート"

    🚧 当社の多言語ドキュメントは現在建設中であり、改善のために一生懸命作業を行っています。ご理解いただきありがとうございます！🙏

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>視聴する:</strong> Ultralytics YOLOタスクの探索：オブジェクト検出、セグメンテーション、トラッキング、ポーズ推定。
</p>

## [検出](detect.md)

検出はYOLOv8がサポートする基本的なタスクです。それは画像やビデオフレーム内のオブジェクトを検出し、周囲に境界ボックスを描くことを含みます。検出されたオブジェクトはその特徴に基づいて異なるカテゴリーに分類されます。YOLOv8は一枚の画像やビデオフレームに複数のオブジェクトを高い精度と速度で検出することができます。

[検出例](detect.md){ .md-button }

## [セグメンテーション](segment.md)

セグメンテーションは、画像の内容に基づいて画像を異なる領域に分割するタスクです。各領域はその内容に基づいてラベルが割り当てられます。このタスクは、画像分割や医療画像処理などのアプリケーションにおいて有用です。YOLOv8はU-Netアーキテクチャのバリエーションを使用してセグメンテーションを実行します。

[セグメンテーション例](segment.md){ .md-button }

## [分類](classify.md)

分類は、画像を異なるカテゴリーに分類するタスクです。YOLOv8は画像の内容に基づいて画像を分類するために使用できます。それはEfficientNetアーキテクチャのバリエーションを使用して分類を実行します。

[分類例](classify.md){ .md-button }

## [ポーズ](pose.md)

ポーズ/キーポイント検出は、画像やビデオフレーム内の特定の点を検出するタスクです。これらの点はキーポイントと呼ばれ、動きやポーズ推定を追跡するために使用されます。YOLOv8は高い精度と速度で画像やビデオフレーム内のキーポイントを検出することができます。

[ポーズ例](pose.md){ .md-button }

## 結論

YOLOv8は、検出、セグメンテーション、分類、キーポイント検出を含む複数のタスクをサポートしています。これらのタスクはそれぞれ異なる目的と用途を持っています。これらのタスクの違いを理解することにより、コンピュータービジョンアプリケーションに適切なタスクを選択することができます。
