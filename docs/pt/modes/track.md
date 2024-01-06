---
comments: true
description: Aprenda a usar o Ultralytics YOLO para rastreamento de objetos em fluxos de vídeo. Guias para usar diferentes rastreadores e personalizar configurações de rastreador.
keywords: Ultralytics, YOLO, rastreamento de objetos, fluxos de vídeo, BoT-SORT, ByteTrack, guia em Python, guia CLI
---

# Rastreamento de Múltiplos Objetos com Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Exemplos de rastreamento de múltiplos objetos">

Rastreamento de objetos no âmbito da análise de vídeo é uma tarefa crucial que não apenas identifica a localização e classe dos objetos dentro do quadro, mas também mantém um ID único para cada objeto detectado à medida que o vídeo avança. As aplicações são ilimitadas — variando desde vigilância e segurança até análises esportivas em tempo real.

## Por Que Escolher Ultralytics YOLO para Rastreamento de Objetos?

A saída dos rastreadores da Ultralytics é consistente com a detecção de objetos padrão, mas com o valor agregado dos IDs dos objetos. Isso facilita o rastreamento de objetos em fluxos de vídeo e a realização de análises subsequentes. Aqui está o porquê de considerar usar Ultralytics YOLO para suas necessidades de rastreamento de objetos:

- **Eficiência:** Processa fluxos de vídeo em tempo real sem comprometer a precisão.
- **Flexibilidade:** Suporta múltiplos algoritmos de rastreamento e configurações.
- **Facilidade de Uso:** Simples API em Python e opções CLI para rápida integração e implantação.
- **Personalização:** Fácil de usar com modelos YOLO treinados personalizados, permitindo integração em aplicações específicas de domínio.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assistir:</strong> Detecção e Rastreamento de Objetos com Ultralytics YOLOv8.
</p>

## Aplicações no Mundo Real

|                                                           Transporte                                                           |                                                            Varejo                                                             |                                                         Aquicultura                                                          |
|:------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|
| ![Rastreamento de Veículos](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![Rastreamento de Pessoas](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![Rastreamento de Peixes](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    Rastreamento de Veículos                                                    |                                                    Rastreamento de Pessoas                                                    |                                                    Rastreamento de Peixes                                                    |

## Características em Destaque

Ultralytics YOLO estende suas funcionalidades de detecção de objetos para fornecer rastreamento de objetos robusto e versátil:

- **Rastreamento em Tempo Real:** Acompanha objetos de forma contínua em vídeos de alta taxa de quadros.
- **Suporte a Múltiplos Rastreadores:** Escolha dentre uma variedade de algoritmos de rastreamento estabelecidos.
- **Configurações de Rastreador Personalizáveis:** Adapte o algoritmo de rastreamento para atender requisitos específicos ajustando vários parâmetros.

## Rastreadores Disponíveis

Ultralytics YOLO suporta os seguintes algoritmos de rastreamento. Eles podem ser ativados passando o respectivo arquivo de configuração YAML, como `tracker=tracker_type.yaml`:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` para ativar este rastreador.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Use `bytetrack.yaml` para ativar este rastreador.

O rastreador padrão é o BoT-SORT.

## Rastreamento

Para executar o rastreador em fluxos de vídeo, use um modelo Detect, Segment ou Pose treinado, como YOLOv8n, YOLOv8n-seg e YOLOv8n-pose.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo oficial ou personalizado
        model = YOLO('yolov8n.pt')  # Carregar um modelo Detect oficial
        model = YOLO('yolov8n-seg.pt')  # Carregar um modelo Segment oficial
        model = YOLO('yolov8n-pose.pt')  # Carregar um modelo Pose oficial
        model = YOLO('caminho/para/melhor.pt')  # Carregar um modelo treinado personalizado

        # Realizar rastreamento com o modelo
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Rastreamento com rastreador padrão
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Rastreamento com o rastreador ByteTrack
        ```

    === "CLI"

        ```bash
        # Realizar rastreamento com vários modelos usando a interface de linha de comando
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo Detect oficial
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo Segment oficial
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo Pose oficial
        yolo track model=caminho/para/melhor.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo treinado personalizado

        # Rastrear usando o rastreador ByteTrack
        yolo track model=caminho/para/melhor.pt tracker="bytetrack.yaml"
        ```

Como pode ser visto no uso acima, o rastreamento está disponível para todos os modelos Detect, Segment e Pose executados em vídeos ou fontes de streaming.

## Configuração

### Argumentos de Rastreamento

A configuração de rastreamento compartilha propriedades com o modo Predict, como `conf`, `iou`, e `show`. Para mais configurações, consulte a página de [Predict](https://docs.ultralytics.com/modes/predict/) model page.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configurar os parâmetros de rastreamento e executar o rastreador
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configurar parâmetros de rastreamento e executar o rastreador usando a interface de linha de comando
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Seleção de Rastreador

A Ultralytics também permite que você use um arquivo de configuração de rastreador modificado. Para fazer isso, simplesmente faça uma cópia de um arquivo de configuração de rastreador (por exemplo, `custom_tracker.yaml`) de [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) e modifique quaisquer configurações (exceto `tracker_type`) conforme suas necessidades.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar o modelo e executar o rastreador com um arquivo de configuração personalizado
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Carregar o modelo e executar o rastreador com um arquivo de configuração personalizado usando a interface de linha de comando
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Para uma lista completa de argumentos de rastreamento, consulte a página [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers).

## Exemplos em Python

### Loop de Persistência de Rastreamentos

Aqui está um script em Python usando OpenCV (`cv2`) e YOLOv8 para executar rastreamento de objetos em quadros de vídeo. Este script ainda pressupõe que você já instalou os pacotes necessários (`opencv-python` e `ultralytics`). O argumento `persist=True` indica ao rastreador que a imagem ou quadro atual é o próximo de uma sequência e que espera rastreamentos da imagem anterior na imagem atual.

!!! Example "Loop de fluxo com rastreamento"

    ```python
    import cv2
    from ultralytics import YOLO

    # Carregar o modelo YOLOv8
    model = YOLO('yolov8n.pt')

    # Abrir o arquivo de vídeo
    video_path = "caminho/para/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Repetir através dos quadros de vídeo
    while cap.isOpened():
        # Ler um quadro do vídeo
        success, frame = cap.read()

        if success:
            # Executar rastreamento YOLOv8 no quadro, persistindo rastreamentos entre quadros
            results = model.track(frame, persist=True)

            # Visualizar os resultados no quadro
            annotated_frame = results[0].plot()

            # Exibir o quadro anotado
            cv2.imshow("Rastreamento YOLOv8", annotated_frame)

            # Interromper o loop se 'q' for pressionado
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Interromper o loop se o fim do vídeo for atingido
            break

    # Liberar o objeto de captura de vídeo e fechar a janela de exibição
    cap.release()
    cv2.destroyAllWindows()
    ```

Note a mudança de `model(frame)` para `model.track(frame)`, que habilita o rastreamento de objetos ao invés de detecção simples. Este script modificado irá executar o rastreador em cada quadro do vídeo, visualizar os resultados e exibi-los em uma janela. O loop pode ser encerrado pressionando 'q'.

## Contribuir com Novos Rastreadores

Você é proficiente em rastreamento de múltiplos objetos e implementou ou adaptou com sucesso um algoritmo de rastreamento com Ultralytics YOLO? Convidamos você a contribuir para nossa seção de Rastreadores em [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Suas aplicações do mundo real e soluções podem ser inestimáveis para usuários trabalhando em tarefas de rastreamento.

Ao contribuir para esta seção, você ajuda a expandir o escopo de soluções de rastreamento disponíveis dentro do framework Ultralytics YOLO, adicionando outra camada de funcionalidade e utilidade para a comunidade.

Para iniciar sua contribuição, por favor, consulte nosso [Guia de Contribuição](https://docs.ultralytics.com/help/contributing) para instruções completas sobre como enviar um Pedido de Pull (PR) 🛠️. Estamos ansiosos para ver o que você traz para a mesa!

Juntos, vamos aprimorar as capacidades de rastreamento do ecossistema Ultralytics YOLO 🙏!
