---
comments: true
description: Da treinamento a rastreamento, aproveite ao máximo o YOLOv8 da Ultralytics. Obtenha insights e exemplos para cada modo suportado, incluindo validação, exportação e benchmarking.
keywords: Ultralytics, YOLOv8, Aprendizado de Máquina, Detecção de Objetos, Treinamento, Validação, Predição, Exportação, Rastreamento, Benchmarking
---

# Modos Ultralytics YOLOv8

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema e integrações do Ultralytics YOLO">

## Introdução

O Ultralytics YOLOv8 não é apenas mais um modelo de detecção de objetos; é um framework versátil projetado para cobrir todo o ciclo de vida dos modelos de aprendizado de máquina — desde a ingestão de dados e treinamento do modelo até a validação, implantação e rastreamento no mundo real. Cada modo serve a um propósito específico e é projetado para oferecer a flexibilidade e eficiência necessárias para diferentes tarefas e casos de uso.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Tutorial dos Modos Ultralytics: Treinar, Validar, Prever, Exportar e Benchmark.
</p>

### Visão Geral dos Modos

Entender os diferentes **modos** que o Ultralytics YOLOv8 suporta é crítico para tirar o máximo proveito de seus modelos:

- **Modo Treino**: Ajuste fino do seu modelo em conjuntos de dados personalizados ou pré-carregados.
- **Modo Validação (Val)**: Um checkpoint pós-treinamento para validar o desempenho do modelo.
- **Modo Predição (Predict)**: Libere o poder preditivo do seu modelo em dados do mundo real.
- **Modo Exportação (Export)**: Prepare seu modelo para implantação em vários formatos.
- **Modo Rastreamento (Track)**: Estenda seu modelo de detecção de objetos para aplicações de rastreamento em tempo real.
- **Modo Benchmarking**: Analise a velocidade e precisão do seu modelo em diversos ambientes de implantação.

Este guia abrangente visa fornecer uma visão geral e insights práticos para cada modo, ajudando você a aproveitar o potencial total do YOLOv8.

## [Treinar](train.md)

O modo Treinar é utilizado para treinar um modelo YOLOv8 em um conjunto de dados personalizado. Neste modo, o modelo é treinado usando o conjunto de dados especificado e os hiperparâmetros escolhidos. O processo de treinamento envolve otimizar os parâmetros do modelo para que ele possa prever com precisão as classes e localizações de objetos em uma imagem.

[Exemplos de Treino](train.md){ .md-button }

## [Validar](val.md)

O modo Validar é utilizado para validar um modelo YOLOv8 após ter sido treinado. Neste modo, o modelo é avaliado em um conjunto de validação para medir sua precisão e desempenho de generalização. Este modo pode ser usado para ajustar os hiperparâmetros do modelo para melhorar seu desempenho.

[Exemplos de Validação](val.md){ .md-button }

## [Prever](predict.md)

O modo Prever é utilizado para fazer previsões usando um modelo YOLOv8 treinado em novas imagens ou vídeos. Neste modo, o modelo é carregado de um arquivo de checkpoint, e o usuário pode fornecer imagens ou vídeos para realizar a inferência. O modelo prevê as classes e localizações dos objetos nas imagens ou vídeos fornecidos.

[Exemplos de Predição](predict.md){ .md-button }

## [Exportar](export.md)

O modo Exportar é utilizado para exportar um modelo YOLOv8 para um formato que possa ser utilizado para implantação. Neste modo, o modelo é convertido para um formato que possa ser utilizado por outras aplicações de software ou dispositivos de hardware. Este modo é útil ao implantar o modelo em ambientes de produção.

[Exemplos de Exportação](export.md){ .md-button }

## [Rastrear](track.md)

O modo Rastrear é utilizado para rastrear objetos em tempo real usando um modelo YOLOv8. Neste modo, o modelo é carregado de um arquivo de checkpoint, e o usuário pode fornecer um fluxo de vídeo ao vivo para realizar o rastreamento de objetos em tempo real. Este modo é útil para aplicações como sistemas de vigilância ou carros autônomos.

[Exemplos de Rastreamento](track.md){ .md-button }

## [Benchmark](benchmark.md)

O modo Benchmark é utilizado para fazer um perfil da velocidade e precisão de vários formatos de exportação para o YOLOv8. Os benchmarks fornecem informações sobre o tamanho do formato exportado, suas métricas `mAP50-95` (para detecção de objetos, segmentação e pose) ou `accuracy_top5` (para classificação), e o tempo de inferência em milissegundos por imagem em diversos formatos de exportação, como ONNX, OpenVINO, TensorRT e outros. Essas informações podem ajudar os usuários a escolher o formato de exportação ótimo para seu caso de uso específico, com base em seus requisitos de velocidade e precisão.

[Exemplos de Benchmark](benchmark.md){ .md-button }
