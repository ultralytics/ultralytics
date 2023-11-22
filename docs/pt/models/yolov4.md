---
comments: true
description: Explore nosso guia detalhado sobre o YOLOv4, um detector de objetos em tempo real de última geração. Entenda seus destaques arquiteturais, recursos inovadores e exemplos de aplicação.
keywords: ultralytics, YOLOv4, detecção de objetos, rede neural, detecção em tempo real, detector de objetos, aprendizado de máquina
---

# YOLOv4: Detecção de Objetos Rápida e Precisa

Bem-vindo à página de documentação do Ultralytics para o YOLOv4, um detector de objetos em tempo real de última geração lançado em 2020 por Alexey Bochkovskiy em [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). O YOLOv4 foi projetado para fornecer o equilíbrio ideal entre velocidade e precisão, tornando-o uma excelente escolha para muitas aplicações.

![Diagrama da arquitetura do YOLOv4](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**Diagrama da arquitetura do YOLOv4**. Mostra o design intricado da rede do YOLOv4, incluindo os componentes backbone, neck e head, bem como suas camadas interconectadas para uma detecção de objetos em tempo real otimizada.

## Introdução

YOLOv4 significa You Only Look Once versão 4. É um modelo de detecção de objetos em tempo real desenvolvido para superar as limitações de versões anteriores do YOLO, como [YOLOv3](yolov3.md) e outros modelos de detecção de objetos. Ao contrário de outros detectores de objetos baseados em redes neurais convolucionais (CNN), o YOLOv4 é aplicável não apenas a sistemas de recomendação, mas também ao gerenciamento de processos independentes e à redução da entrada humana. Sua operação em unidades de processamento gráfico (GPUs) convencionais permite o uso em massa a um preço acessível, e foi projetado para funcionar em tempo real em uma GPU convencional, exigindo apenas uma GPU para treinamento.

## Arquitetura

O YOLOv4 faz uso de várias características inovadoras que trabalham juntas para otimizar seu desempenho. Estas incluem Conexões Residuais Ponderadas (WRC), Conexões Parciais Cruzadas de Estágio (CSP), Normalização Cruzada em Mini Lote (CmBN), Treinamento Autoadversário (SAT), Ativação Mish, Aumento de Dados Mosaic, Regularização DropBlock e Perda CIoU. Essas características são combinadas para obter resultados de última geração.

Um detector de objetos típico é composto por várias partes, incluindo a entrada, o backbone, o neck e o head. O backbone do YOLOv4 é pré-treinado no ImageNet e é usado para prever as classes e caixas delimitadoras dos objetos. O backbone pode ser de vários modelos, incluindo VGG, ResNet, ResNeXt ou DenseNet. A parte neck do detector é usada para coletar mapas de características de diferentes estágios e geralmente inclui várias caminhadas bottom-up e várias caminhadas top-down. A parte head é responsável por fazer as detecções e classificações finais dos objetos.

## Bag of Freebies

O YOLOv4 também faz uso de métodos conhecidos como "bag of freebies" (saco de brindes), que são técnicas que melhoram a precisão do modelo durante o treinamento sem aumentar o custo da inferência. O aumento de dados é uma técnica comum de "bag of freebies" usada na detecção de objetos, que aumenta a variabilidade das imagens de entrada para melhorar a robustez do modelo. Alguns exemplos de aumento de dados incluem distorções fotométricas (ajustando o brilho, contraste, matiz, saturação e ruído de uma imagem) e distorções geométricas (adicionando dimensionamento aleatório, recorte, espelhamento e rotação). Essas técnicas ajudam o modelo a generalizar melhor para diferentes tipos de imagens.

## Recursos e Desempenho

O YOLOv4 foi projetado para oferecer velocidade e precisão ideais na detecção de objetos. A arquitetura do YOLOv4 inclui o CSPDarknet53 como o backbone, o PANet como o neck e o YOLOv3 como a cabeça de detecção. Esse design permite que o YOLOv4 realize detecção de objetos em uma velocidade impressionante, tornando-o adequado para aplicações em tempo real. O YOLOv4 também se destaca em termos de precisão, alcançando resultados de última geração em benchmarks de detecção de objetos.

## Exemplos de Uso

No momento da escrita, o Ultralytics não oferece suporte a modelos YOLOv4. Portanto, os usuários interessados em usar o YOLOv4 deverão consultar diretamente o repositório YOLOv4 no GitHub para instruções de instalação e uso.

Aqui está uma breve visão geral das etapas típicas que você pode seguir para usar o YOLOv4:

1. Visite o repositório YOLOv4 no GitHub: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Siga as instruções fornecidas no arquivo README para a instalação. Isso geralmente envolve clonar o repositório, instalar as dependências necessárias e configurar as variáveis de ambiente necessárias.

3. Uma vez que a instalação esteja completa, você pode treinar e usar o modelo de acordo com as instruções de uso fornecidas no repositório. Isso geralmente envolve a preparação do seu conjunto de dados, a configuração dos parâmetros do modelo, o treinamento do modelo e, em seguida, o uso do modelo treinado para realizar a detecção de objetos.

Observe que as etapas específicas podem variar dependendo do seu caso de uso específico e do estado atual do repositório YOLOv4. Portanto, é altamente recomendável consultar diretamente as instruções fornecidas no repositório YOLOv4 do GitHub.

Lamentamos qualquer inconveniente que isso possa causar e nos esforçaremos para atualizar este documento com exemplos de uso para o Ultralytics assim que o suporte para o YOLOv4 for implementado.

## Conclusão

O YOLOv4 é um modelo poderoso e eficiente de detecção de objetos que oferece um equilíbrio entre velocidade e precisão. O uso de recursos exclusivos e técnicas "Bag of Freebies" durante o treinamento permite que ele tenha um excelente desempenho em tarefas de detecção de objetos em tempo real. O YOLOv4 pode ser treinado e usado por qualquer pessoa com uma GPU convencional, tornando-o acessível e prático para uma ampla variedade de aplicações.

## Referências e Agradecimentos

Gostaríamos de agradecer aos autores do YOLOv4 por suas contribuições significativas no campo da detecção de objetos em tempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

O artigo original do YOLOv4 pode ser encontrado no [arXiv](https://arxiv.org/pdf/2004.10934.pdf). Os autores disponibilizaram seu trabalho publicamente, e o código pode ser acessado no [GitHub](https://github.com/AlexeyAB/darknet). Agradecemos seus esforços em avançar o campo e tornar seu trabalho acessível à comunidade em geral.
