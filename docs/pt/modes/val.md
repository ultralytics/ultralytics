---
comments: true
description: Guia para Validação de Modelos YOLOv8. Aprenda como avaliar o desempenho dos seus modelos YOLO utilizando configurações e métricas de validação com exemplos em Python e CLI.
keywords: Ultralytics, Documentação YOLO, YOLOv8, validação, avaliação de modelo, hiperparâmetros, precisão, métricas, Python, CLI
---

# Validação de Modelos com Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema e integrações do Ultralytics YOLO">

## Introdução

A validação é um passo crítico no pipeline de aprendizado de máquina, permitindo que você avalie a qualidade dos seus modelos treinados. O modo Val no Ultralytics YOLOv8 fornece um robusto conjunto de ferramentas e métricas para avaliar o desempenho dos seus modelos de detecção de objetos. Este guia serve como um recurso completo para entender como usar efetivamente o modo Val para garantir que seus modelos sejam precisos e confiáveis.

## Por Que Validar com o Ultralytics YOLO?

Aqui estão as vantagens de usar o modo Val no YOLOv8:

- **Precisão:** Obtenha métricas precisas como mAP50, mAP75 e mAP50-95 para avaliar seu modelo de forma abrangente.
- **Conveniência:** Utilize recursos integrados que lembram as configurações de treinamento, simplificando o processo de validação.
- **Flexibilidade:** Valide seu modelo com os mesmos ou diferentes conjuntos de dados e tamanhos de imagem.
- **Ajuste de Hiperparâmetros:** Utilize as métricas de validação para refinar seu modelo e obter um desempenho melhor.

### Principais Recursos do Modo Val

Estas são as funcionalidades notáveis oferecidas pelo modo Val do YOLOv8:

- **Configurações Automatizadas:** Os modelos lembram suas configurações de treinamento para validação direta.
- **Suporte Multi-Métrico:** Avalie seu modelo com base em uma variedade de métricas de precisão.
- **API em Python e CLI:** Escolha entre a interface de linha de comando ou API em Python com base na sua preferência de validação.
- **Compatibilidade de Dados:** Funciona perfeitamente com conjuntos de dados usados durante a fase de treinamento, bem como conjuntos de dados personalizados.

!!! Tip "Dica"

    * Os modelos YOLOv8 lembram automaticamente suas configurações de treinamento, então você pode validar um modelo no mesmo tamanho de imagem e no conjunto de dados original facilmente com apenas `yolo val model=yolov8n.pt` ou `model('yolov8n.pt').val()`

## Exemplos de Uso

Validar a precisão do modelo YOLOv8n treinado no conjunto de dados COCO128. Nenhum argumento precisa ser passado, pois o `model` retém os dados de treinamento e argumentos como atributos do modelo. Veja a seção de Argumentos abaixo para uma lista completa dos argumentos de exportação.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo personalizado

        # Validar o modelo
        metrics = model.val()  # nenhum argumento necessário, conjunto de dados e configurações lembrados
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # uma lista contém map50-95 de cada categoria
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # validar modelo oficial
        yolo detect val model=path/to/best.pt  # validar modelo personalizado
        ```

## Argumentos

As configurações de validação para os modelos YOLO referem-se aos vários hiperparâmetros e configurações usados para avaliar o desempenho do modelo em um conjunto de dados de validação. Essas configurações podem afetar o desempenho, velocidade e precisão do modelo. Algumas configurações comuns de validação do YOLO incluem o tamanho do lote, a frequência com que a validação é realizada durante o treinamento e as métricas usadas para avaliar o desempenho do modelo. Outros fatores que podem afetar o processo de validação incluem o tamanho e a composição do conjunto de dados de validação e a tarefa específica para a qual o modelo está sendo usado. É importante ajustar e experimentar cuidadosamente essas configurações para garantir que o modelo apresente um bom desempenho no conjunto de dados de validação e para detectar e prevenir o sobreajuste.

| Chave         | Valor   | Descrição                                                                         |
|---------------|---------|-----------------------------------------------------------------------------------|
| `data`        | `None`  | caminho para o arquivo de dados, ex. coco128.yaml                                 |
| `imgsz`       | `640`   | tamanho das imagens de entrada como inteiro                                       |
| `batch`       | `16`    | número de imagens por lote (-1 para AutoBatch)                                    |
| `save_json`   | `False` | salvar resultados em arquivo JSON                                                 |
| `save_hybrid` | `False` | salvar versão híbrida das etiquetas (etiquetas + previsões adicionais)            |
| `conf`        | `0.001` | limite de confiança do objeto para detecção                                       |
| `iou`         | `0.6`   | limiar de interseção sobre união (IoU) para NMS                                   |
| `max_det`     | `300`   | número máximo de detecções por imagem                                             |
| `half`        | `True`  | usar precisão meia (FP16)                                                         |
| `device`      | `None`  | dispositivo para execução, ex. dispositivo cuda=0/1/2/3 ou device=cpu             |
| `dnn`         | `False` | usar OpenCV DNN para inferência ONNX                                              |
| `plots`       | `False` | mostrar gráficos durante o treinamento                                            |
| `rect`        | `False` | val retangular com cada lote colado para minimizar o preenchimento                |
| `split`       | `val`   | divisão do conjunto de dados para usar na validação, ex. 'val', 'test' ou 'train' |
|
