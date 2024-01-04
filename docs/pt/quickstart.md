---
comments: true
description: Explore os diversos métodos para instalar o Ultralytics usando pip, conda, git e Docker. Aprenda a usar o Ultralytics com a interface de linha de comando ou dentro dos seus projetos Python.
keywords: Instalação do Ultralytics, pip install Ultralytics, Docker install Ultralytics, interface de linha de comando do Ultralytics, interface Python do Ultralytics
---

## Instalação do Ultralytics

O Ultralytics oferece diversos métodos de instalação, incluindo pip, conda e Docker. Instale o YOLOv8 através do pacote `ultralytics` pip para a versão estável mais recente ou clonando o [repositório GitHub do Ultralytics](https://github.com/ultralytics/ultralytics) para obter a versão mais atualizada. O Docker pode ser usado para executar o pacote em um contêiner isolado, evitando a instalação local.

!!! Example "Instalar"

    === "Pip install (recomendado)"
        Instale o pacote `ultralytics` usando pip, ou atualize uma instalação existente executando `pip install -U ultralytics`. Visite o Índice de Pacotes Python (PyPI) para mais detalhes sobre o pacote `ultralytics`: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Instalar o pacote ultralytics do PyPI
        pip install ultralytics
        ```

        Você também pode instalar o pacote `ultralytics` diretamente do [repositório](https://github.com/ultralytics/ultralytics) GitHub. Isso pode ser útil se você desejar a versão de desenvolvimento mais recente. Certifique-se de ter a ferramenta de linha de comando Git instalada no seu sistema. O comando `@main` instala a branch `main` e pode ser modificado para outra branch, ou seja, `@my-branch`, ou removido completamente para padrão na branch `main`.

        ```bash
        # Instalar o pacote ultralytics do GitHub
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Conda install"
        Conda é um gerenciador de pacotes alternativo ao pip que também pode ser usado para instalação. Visite Anaconda para mais detalhes em [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). O repositório de feedstock do Ultralytics para atualizar o pacote conda está em [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).


        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Instalar o pacote ultralytics usando conda
        conda install -c conda-forge ultralytics
        ```

        !!! Note "Nota"

            Se você está instalando em um ambiente CUDA a prática recomendada é instalar `ultralytics`, `pytorch` e `pytorch-cuda` no mesmo comando para permitir que o gerenciador de pacotes conda resolva quaisquer conflitos, ou instalar `pytorch-cuda` por último para permitir que ele substitua o pacote específico para CPU `pytorch`, se necessário.
            ```bash
            # Instalar todos os pacotes juntos usando conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Imagem Docker Conda

        As imagens Docker Conda do Ultralytics também estão disponíveis em [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics). Estas imagens são baseadas em [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) e são um modo simples de começar a usar `ultralytics` em um ambiente Conda.

        ```bash
        # Definir o nome da imagem como uma variável
        t=ultralytics/ultralytics:latest-conda

        # Puxar a imagem mais recente do ultralytics do Docker Hub
        sudo docker pull $t

        # Executar a imagem ultralytics em um contêiner com suporte a GPU
        sudo docker run -it --ipc=host --gpus all $t  # todas as GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # especificar GPUs
        ```

    === "Git clone"
        Clone o repositório `ultralytics` se você está interessado em contribuir para o desenvolvimento ou deseja experimentar com o código-fonte mais recente. Após clonar, navegue até o diretório e instale o pacote em modo editável `-e` usando pip.
        ```bash
        # Clonar o repositório ultralytics
        git clone https://github.com/ultralytics/ultralytics

        # Navegar para o diretório clonado
        cd ultralytics

        # Instalar o pacote em modo editável para desenvolvimento
        pip install -e .
        ```

Veja o arquivo [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) do `ultralytics` para uma lista de dependências. Note que todos os exemplos acima instalam todas as dependências necessárias.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! Tip "Dica"

    Os requisitos do PyTorch variam pelo sistema operacional e pelos requisitos de CUDA, então é recomendado instalar o PyTorch primeiro seguindo as instruções em [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="Instruções de Instalação do PyTorch" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Use o Ultralytics com CLI

A interface de linha de comando (CLI) do Ultralytics permite comandos simples de uma única linha sem a necessidade de um ambiente Python. O CLI não requer personalização ou código Python. Você pode simplesmente rodar todas as tarefas do terminal com o comando `yolo`. Confira o [Guia CLI](/../usage/cli.md) para aprender mais sobre o uso do YOLOv8 pela linha de comando.

!!! Example "Exemplo"

    === "Sintaxe"

        Os comandos `yolo` do Ultralytics usam a seguinte sintaxe:
        ```bash
        yolo TAREFA MODO ARGUMENTOS

        Onde   TAREFA (opcional) é um entre [detect, segment, classify]
                MODO (obrigatório) é um entre [train, val, predict, export, track]
                ARGUMENTOS (opcional) são qualquer número de pares personalizados 'arg=valor' como 'imgsz=320' que substituem os padrões.
        ```
        Veja todos os ARGUMENTOS no guia completo de [Configuração](/../usage/cfg.md) ou com `yolo cfg`

    === "Train"

        Treinar um modelo de detecção por 10 épocas com uma taxa de aprendizado inicial de 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Prever um vídeo do YouTube usando um modelo de segmentação pré-treinado com tamanho de imagem 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Validar um modelo de detecção pré-treinado com tamanho de lote 1 e tamanho de imagem 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        Exportar um modelo de classificação YOLOv8n para formato ONNX com tamanho de imagem 224 por 128 (nenhuma TAREFA necessária)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Executar comandos especiais para ver versão, visualizar configurações, rodar verificações e mais:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "Aviso"

    Argumentos devem ser passados como pares `arg=valor`, separados por um sinal de igual `=` e delimitados por espaços ` ` entre pares. Não use prefixos de argumentos `--` ou vírgulas `,` entre os argumentos.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[Guia CLI](/../usage/cli.md){ .md-button }

## Use o Ultralytics com Python

A interface Python do YOLOv8 permite uma integração tranquila em seus projetos Python, tornando fácil carregar, executar e processar a saída do modelo. Projetada com simplicidade e facilidade de uso em mente, a interface Python permite que os usuários implementem rapidamente detecção de objetos, segmentação e classificação em seus projetos. Isto torna a interface Python do YOLOv8 uma ferramenta inestimável para qualquer pessoa buscando incorporar essas funcionalidades em seus projetos Python.

Por exemplo, os usuários podem carregar um modelo, treiná-lo, avaliar o seu desempenho em um conjunto de validação e até exportá-lo para o formato ONNX com apenas algumas linhas de código. Confira o [Guia Python](/../usage/python.md) para aprender mais sobre o uso do YOLOv8 dentro dos seus projetos Python.

!!! Example "Exemplo"

    ```python
    from ultralytics import YOLO

    # Criar um novo modelo YOLO do zero
    model = YOLO('yolov8n.yaml')

    # Carregar um modelo YOLO pré-treinado (recomendado para treinamento)
    model = YOLO('yolov8n.pt')

    # Treinar o modelo usando o conjunto de dados 'coco128.yaml' por 3 épocas
    results = model.train(data='coco128.yaml', epochs=3)

    # Avaliar o desempenho do modelo no conjunto de validação
    results = model.val()

    # Realizar detecção de objetos em uma imagem usando o modelo
    results = model('https://ultralytics.com/images/bus.jpg')

    # Exportar o modelo para formato ONNX
    success = model.export(format='onnx')
    ```

[Guia Python](/../usage/python.md){.md-button .md-button--primary}
