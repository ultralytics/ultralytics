---
comments: true
description: Descubra como usar o modo predict do YOLOv8 para diversas tarefas. Aprenda sobre diferentes fontes de inferência, como imagens, vídeos e formatos de dados.
keywords: Ultralytics, YOLOv8, modo predict, fontes de inferência, tarefas de previsão, modo de streaming, processamento de imagens, processamento de vídeo, aprendizado de máquina, IA
---

# Predição de Modelo com Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema e integrações do Ultralytics YOLO">

## Introdução

No mundo do aprendizado de máquina e visão computacional, o processo de fazer sentido a partir de dados visuais é chamado de 'inferência' ou 'predição'. O Ultralytics YOLOv8 oferece um recurso poderoso conhecido como **modo predict** que é personalizado para inferência em tempo real de alto desempenho em uma ampla gama de fontes de dados.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Como Extrair as Saídas do Modelo Ultralytics YOLOv8 para Projetos Personalizados.
</p>

## Aplicações no Mundo Real

|                                                                   Manufatura                                                                    |                                                               Esportes                                                               |                                                             Segurança                                                              |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------:|
| ![Detecção de Peças de Reposição de Veículo](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Detecção de Jogador de Futebol](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![Detecção de Queda de Pessoas](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                    Detecção de Peças de Reposição de Veículo                                                    |                                                    Detecção de Jogador de Futebol                                                    |                                                    Detecção de Queda de Pessoas                                                    |

## Por Que Usar o Ultralytics YOLO para Inferência?

Aqui está o porquê de você considerar o modo predict do YOLOv8 para suas diversas necessidades de inferência:

- **Versatilidade:** Capaz de fazer inferências em imagens, vídeos e até transmissões ao vivo.
- **Desempenho:** Projetado para processamento em tempo real e de alta velocidade sem sacrificar a precisão.
- **Facilidade de Uso:** Interfaces Python e CLI intuitivas para implantação e testes rápidos.
- **Altamente Customizável:** Várias configurações e parâmetros para ajustar o comportamento de inferência do modelo de acordo com suas necessidades específicas.

### Recursos Chave do Modo Predict

O modo predict do YOLOv8 é projetado para ser robusto e versátil, apresentando:

- **Compatibilidade com Múltiplas Fontes de Dados:** Se seus dados estão na forma de imagens individuais, uma coleção de imagens, arquivos de vídeo ou transmissões de vídeo em tempo real, o modo predict atende a todas as necessidades.
- **Modo de Streaming:** Use o recurso de streaming para gerar um gerador eficiente de memória de objetos `Results`. Ative isso definindo `stream=True` no método de chamada do preditor.
- **Processamento em Lote:** A capacidade de processar várias imagens ou quadros de vídeo em um único lote, acelerando ainda mais o tempo de inferência.
- **Integração Amigável:** Integração fácil com pipelines de dados existentes e outros componentes de software, graças à sua API flexível.

Os modelos Ultralytics YOLO retornam ou uma lista de objetos `Results` em Python, ou um gerador em Python eficiente de memória de objetos `Results` quando `stream=True` é passado para o modelo durante a inferência:

!!! Example "Predict"

    === "Retorna uma lista com `stream=False`"
        ```python
        from ultralytics import YOLO

        # Carrega um modelo
        model = YOLO('yolov8n.pt')  # modelo YOLOv8n pré-treinado

        # Executa a inferência em lote em uma lista de imagens
        results = model(['im1.jpg', 'im2.jpg'])  # retorna uma lista de objetos Results

        # Processa a lista de resultados
        for result in results:
            boxes = result.boxes  # Objeto Boxes para saídas de bbox
            masks = result.masks  # Objeto Masks para saídas de máscaras de segmentação
            keypoints = result.keypoints  # Objeto Keypoints para saídas de pose
            probs = result.probs  # Objeto Probs para saídas de classificação
        ```

    === "Retorna um gerador com `stream=True`"
        ```python
        from ultralytics import YOLO

        # Carrega um modelo
        model = YOLO('yolov8n.pt')  # modelo YOLOv8n pré-treinado

        # Executa a inferência em lote em uma lista de imagens
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # retorna um gerador de objetos Results

        # Processa o gerador de resultados
        for result in results:
            boxes = result.boxes  # Objeto Boxes para saídas de bbox
            masks = result.masks  # Objeto Masks para saídas de máscaras de segmentação
            keypoints = result.keypoints  # Objeto Keypoints para saídas de pose
            probs = result.probs  # Objeto Probs para saídas de classificação
        ```

## Fontes de Inferência

O YOLOv8 pode processar diferentes tipos de fontes de entrada para inferência, conforme mostrado na tabela abaixo. As fontes incluem imagens estáticas, transmissões de vídeo e vários formatos de dados. A tabela também indica se cada fonte pode ser usada no modo de streaming com o argumento `stream=True` ✅. O modo de streaming é benéfico para processar vídeos ou transmissões ao vivo, pois cria um gerador de resultados em vez de carregar todos os quadros na memória.

!!! Tip "Dica"

    Use `stream=True` para processar vídeos longos ou grandes conjuntos de dados para gerenciar a memória de forma eficiente. Quando `stream=False`, os resultados de todos os quadros ou pontos de dados são armazenados na memória, o que pode aumentar rapidamente e causar erros de falta de memória para grandes entradas. Em contraste, `stream=True` utiliza um gerador, que mantém apenas os resultados do quadro atual ou ponto de dados na memória, reduzindo significativamente o consumo de memória e prevenindo problemas de falta dela.

| Fonte           | Argumento                                  | Tipo            | Notas                                                                                                                   |
|-----------------|--------------------------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------|
| imagem          | `'image.jpg'`                              | `str` ou `Path` | Arquivo de imagem único.                                                                                                |
| URL             | `'https://ultralytics.com/images/bus.jpg'` | `str`           | URL para uma imagem.                                                                                                    |
| captura de tela | `'screen'`                                 | `str`           | Captura uma captura de tela.                                                                                            |
| PIL             | `Image.open('im.jpg')`                     | `PIL.Image`     | Formato HWC com canais RGB.                                                                                             |
| OpenCV          | `cv2.imread('im.jpg')`                     | `np.ndarray`    | Formato HWC com canais BGR `uint8 (0-255)`.                                                                             |
| numpy           | `np.zeros((640,1280,3))`                   | `np.ndarray`    | Formato HWC com canais BGR `uint8 (0-255)`.                                                                             |
| torch           | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | Formato BCHW com canais RGB `float32 (0.0-1.0)`.                                                                        |
| CSV             | `'sources.csv'`                            | `str` ou `Path` | Arquivo CSV contendo caminhos para imagens, vídeos ou diretórios.                                                       |
| vídeo ✅         | `'video.mp4'`                              | `str` ou `Path` | Arquivo de vídeo em formatos como MP4, AVI, etc.                                                                        |
| diretório ✅     | `'path/'`                                  | `str` ou `Path` | Caminho para um diretório contendo imagens ou vídeos.                                                                   |
| glob ✅          | `'path/*.jpg'`                             | `str`           | Padrão glob para combinar vários arquivos. Use o caractere `*` como curinga.                                            |
| YouTube ✅       | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | URL para um vídeo do YouTube.                                                                                           |
| stream ✅        | `'rtsp://example.com/media.mp4'`           | `str`           | URL para protocolos de streaming como RTSP, RTMP, TCP ou um endereço IP.                                                |
| multi-stream ✅  | `'list.streams'`                           | `str` ou `Path` | Arquivo de texto `*.streams` com uma URL de stream por linha, ou seja, 8 streams serão executados em lote de tamanho 8. |

Abaixo estão exemplos de código para usar cada tipo de fonte:

!!! Example "Fontes de previsão"

    === "imagem"
        Executa a inferência em um arquivo de imagem.
        ```python
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Define o caminho para o arquivo de imagem
        source = 'caminho/para/imagem.jpg'

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "captura de tela"
        Executa a inferência no conteúdo atual da tela como uma captura de tela.
        ```python
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Define a captura de tela atual como fonte
        source = 'screen'

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "URL"
        Executa a inferência em uma imagem ou vídeo hospedado remotamente via URL.
        ```python
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Define a URL remota da imagem ou vídeo
        source = 'https://ultralytics.com/images/bus.jpg'

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "PIL"
        Executa a inferência em uma imagem aberta com a Biblioteca de Imagens do Python (PIL).
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Abre uma imagem usando PIL
        source = Image.open('caminho/para/imagem.jpg')

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "OpenCV"
        Executa a inferência em uma imagem lida com OpenCV.
        ```python
        import cv2
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Lê uma imagem usando OpenCV
        source = cv2.imread('caminho/para/imagem.jpg')

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "numpy"
        Executa a inferência em uma imagem representada como um array numpy.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Cria um array random de numpy com forma HWC (640, 640, 3) com valores no intervalo [0, 255] e tipo uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```

    === "torch"
        Executa a inferência em uma imagem representada como um tensor PyTorch.
        ```python
        import torch
        from ultralytics import YOLO

        # Carrega um modelo YOLOv8n pré-treinado
        model = YOLO('yolov8n.pt')

        # Cria um tensor random de torch com forma BCHW (1, 3, 640, 640) com valores no intervalo [0, 1] e tipo float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # Executa a inferência na fonte
        results = model(source)  # lista de objetos Results
        ```
