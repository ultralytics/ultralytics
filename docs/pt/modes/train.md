---
comments: true
description: Guia passo a passo para treinar modelos YOLOv8 com a YOLO da Ultralytics, incluindo exemplos de treinamento com uma única GPU e múltiplas GPUs
keywords: Ultralytics, YOLOv8, YOLO, detecção de objetos, modo de treino, conjunto de dados personalizado, treinamento com GPU, multi-GPU, hiperparâmetros, exemplos de CLI, exemplos em Python
---

# Treinamento de Modelos com a YOLO da Ultralytics

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema e integrações da YOLO da Ultralytics">

## Introdução

O treinamento de um modelo de aprendizado profundo envolve fornecer dados e ajustar seus parâmetros para que ele possa fazer previsões precisas. O modo de treino na YOLOv8 da Ultralytics é projetado para um treinamento eficaz e eficiente de modelos de detecção de objetos, aproveitando totalmente as capacidades do hardware moderno. Este guia visa cobrir todos os detalhes que você precisa para começar a treinar seus próprios modelos usando o robusto conjunto de recursos da YOLOv8.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Como Treinar um modelo YOLOv8 no Seu Conjunto de Dados Personalizado no Google Colab.
</p>

## Por Que Escolher a YOLO da Ultralytics para Treinamento?

Aqui estão algumas razões convincentes para optar pelo modo de Treino da YOLOv8:

- **Eficiência:** Aproveite ao máximo seu hardware, seja em um setup com uma única GPU ou expandindo para múltiplas GPUs.
- **Versatilidade:** Treine em conjuntos de dados personalizados, além dos já disponíveis, como COCO, VOC e ImageNet.
- **Facilidade de Uso:** Interfaces de linha de comando (CLI) e em Python simples, porém poderosas, para uma experiência de treinamento direta.
- **Flexibilidade de Hiperparâmetros:** Uma ampla gama de hiperparâmetros personalizáveis para ajustar o desempenho do modelo.

### Principais Recursos do Modo de Treino

Os seguintes são alguns recursos notáveis ​​do modo de Treino da YOLOv8:

- **Download Automático de Datasets:** Datasets padrões como COCO, VOC e ImageNet são baixados automaticamente na primeira utilização.
- **Suporte a Multi-GPU:** Escalone seus esforços de treinamento de maneira uniforme entre várias GPUs para acelerar o processo.
- **Configuração de Hiperparâmetros:** Opção de modificar hiperparâmetros através de arquivos de configuração YAML ou argumentos de CLI.
- **Visualização e Monitoramento:** Acompanhamento em tempo real das métricas de treinamento e visualização do processo de aprendizagem para obter melhores insights.

!!! Tip "Dica"

    * Conjuntos de dados YOLOv8 como COCO, VOC, ImageNet e muitos outros são baixados automaticamente na primeira utilização, ou seja, `yolo train data=coco.yaml`

## Exemplos de Uso

Treine o YOLOv8n no conjunto de dados COCO128 por 100 épocas com tamanho de imagem de 640. O dispositivo de treinamento pode ser especificado usando o argumento `device`. Se nenhum argumento for passado, a GPU `device=0` será usado se disponível, caso contrário, `device=cpu` será usado. Veja a seção Argumentos abaixo para uma lista completa dos argumentos de treinamento.

!!! Example "Exemplo de Treinamento em Uma Única GPU e CPU"

    O dispositivo é determinado automaticamente. Se uma GPU estiver disponível, ela será usada, caso contrário, o treinamento começará na CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.yaml')  # construir um novo modelo a partir do YAML
        model = YOLO('yolov8n.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construir a partir do YAML e transferir pesos

        # Treinar o modelo
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Construir um novo modelo a partir do YAML e começar o treinamento do zero
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Começar o treinamento a partir de um modelo *.pt pré-treinado
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construir um novo modelo a partir do YAML, transferir pesos pré-treinados para ele e começar o treinamento
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Treinamento com Multi-GPU

O treinamento com múltiplas GPUs permite uma utilização mais eficiente dos recursos de hardware disponíveis, distribuindo a carga de treinamento entre várias GPUs. Esse recurso está disponível por meio da API do Python e da interface de linha de comando. Para habilitar o treinamento com várias GPUs, especifique os IDs dos dispositivos de GPU que deseja usar.

!!! Example "Exemplo de Treinamento com Multi-GPU"

    Para treinar com 2 GPUs, dispositivos CUDA 0 e 1 use os seguintes comandos. Expanda para GPUs adicionais conforme necessário.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)

        # Treinar o modelo com 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Começar o treinamento a partir de um modelo *.pt pré-treinado usando as GPUs 0 e 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Treinamento com Apple M1 e M2 MPS

Com a integração do suporte para os chips Apple M1 e M2 nos modelos YOLO da Ultralytics, agora é possível treinar seus modelos em dispositivos que utilizam o poderoso framework Metal Performance Shaders (MPS). O MPS oferece uma forma de alto desempenho de executar tarefas de computação e processamento de imagens no silício personalizado da Apple.

Para habilitar o treinamento nos chips Apple M1 e M2, você deve especificar 'mps' como seu dispositivo ao iniciar o processo de treinamento. Abaixo está um exemplo de como você pode fazer isso em Python e via linha de comando:

!!! Example "Exemplo de Treinamento com MPS"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)

        # Treinar o modelo com MPS
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Começar o treinamento a partir de um modelo *.pt pré-treinado usando MPS
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

Ao aproveitar o poder computacional dos chips M1/M2, isso possibilita o processamento mais eficiente das tarefas de treinamento. Para orientações mais detalhadas e opções avançadas de configuração, consulte a [documentação do PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html).

## Registro de Logs

Ao treinar um modelo YOLOv8, você pode achar valioso acompanhar o desempenho do modelo ao longo do tempo. É aqui que o registro de logs se torna útil. O YOLO da Ultralytics oferece suporte para três tipos de loggers - Comet, ClearML e TensorBoard.

Para usar um logger, selecione-o no menu suspenso no trecho de código acima e execute-o. O logger escolhido será instalado e inicializado.

### Comet

[Comet](https://www.comet.ml/site/) é uma plataforma que permite a cientistas de dados e desenvolvedores rastrear, comparar, explicar e otimizar experimentos e modelos. Oferece funcionalidades como métricas em tempo real, diffs de código e acompanhamento de hiperparâmetros.

Para usar o Comet:

!!! Example "Exemplo"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Lembre-se de fazer login na sua conta Comet no site deles e obter sua chave de API. Você precisará adicionar isso às suas variáveis de ambiente ou ao seu script para registrar seus experimentos.

### ClearML

[ClearML](https://www.clear.ml/) é uma plataforma de código aberto que automatiza o rastreamento de experimentos e ajuda com o compartilhamento eficiente de recursos. É projetada para ajudar as equipes a gerenciar, executar e reproduzir seus trabalhos de ML de maneira mais eficiente.

Para usar o ClearML:

!!! Example "Exemplo"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

Após executar este script, você precisará fazer login na sua conta ClearML no navegador e autenticar sua sessão.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) é um kit de ferramentas de visualização para TensorFlow. Permite visualizar o seu gráfico TensorFlow, plotar métricas quantitativas sobre a execução do seu gráfico e mostrar dados adicionais como imagens que passam por ele.

Para usar o TensorBoard em [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb):

!!! Example "Exemplo"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # substitua pelo diretório 'runs'
        ```

Para usar o TensorBoard localmente, execute o comando abaixo e veja os resultados em http://localhost:6006/:

!!! Example "Exemplo"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # substitua pelo diretório 'runs'
        ```

Isso irá carregar o TensorBoard e direcioná-lo para o diretório onde seus logs de treinamento estão salvos.

Depois de configurar o seu logger, você pode então prosseguir com o treinamento do seu modelo. Todas as métricas de treinamento serão registradas automaticamente na sua plataforma escolhida, e você pode acessar esses logs para monitorar o desempenho do seu modelo ao longo do tempo, comparar diferentes modelos e identificar áreas para melhoria.
