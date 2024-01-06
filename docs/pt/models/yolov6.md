---
comments: true
description: Explore Meituan YOLOv6, um modelo avançado de detecção de objetos que alcança um equilíbrio entre velocidade e precisão. Saiba mais sobre suas características, modelos pré-treinados e uso em Python.
keywords: Meituan YOLOv6, detecção de objetos, Ultralytics, documentação YOLOv6, Concatenação Bidirecional, Treinamento Assistido por Âncora, modelos pré-treinados, aplicações em tempo real
---

# Meituan YOLOv6

## Visão Geral

O Meituan YOLOv6 é um detector de objetos de ponta que oferece um equilíbrio notável entre velocidade e precisão, tornando-se uma escolha popular para aplicações em tempo real. Este modelo apresenta várias melhorias em sua arquitetura e esquema de treinamento, incluindo a implementação de um módulo de Concatenação Bidirecional (BiC), uma estratégia de treinamento assistido por âncora (AAT) e um design aprimorado de espinha dorsal e pescoço para obter precisão de última geração no conjunto de dados COCO.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![Modelo exemplo de imagem](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**Visão geral do YOLOv6.** Diagrama da arquitetura do modelo mostrando os componentes de rede redesenhados e as estratégias de treinamento que levaram a melhorias significativas no desempenho. (a) O pescoço do YOLOv6 (N e S são mostrados). RepBlocks é substituída por CSPStackRep para M/L. (b) A estrutura de um módulo BiC. (c) Um bloco SimCSPSPPF. ([fonte](https://arxiv.org/pdf/2301.05586.pdf)).

### Principais Características

- **Módulo de Concatenação Bidirecional (BiC):** O YOLOv6 introduz um módulo BiC no pescoço do detector, aprimorando os sinais de localização e oferecendo ganhos de desempenho com uma degradação de velocidade insignificante.
- **Estratégia de Treinamento Assistido por Âncora (AAT):** Este modelo propõe AAT para aproveitar os benefícios dos paradigmas baseados em âncoras e sem âncoras sem comprometer a eficiência da inferência.
- **Design de Espinha Dorsal e Pescoço Aprimorado:** Ao aprofundar o YOLOv6 para incluir mais uma etapa na espinha dorsal e no pescoço, este modelo alcança desempenho de última geração no conjunto de dados COCO com entrada de alta resolução.
- **Estratégia de Auto-Destilação:** Uma nova estratégia de auto-destilação é implementada para aumentar o desempenho de modelos menores do YOLOv6, aprimorando o ramo auxiliar de regressão durante o treinamento e removendo-o durante a inferência para evitar uma queda significativa na velocidade.

## Métricas de Desempenho

O YOLOv6 fornece vários modelos pré-treinados com diferentes escalas:

- YOLOv6-N: 37,5% AP na val2017 do COCO a 1187 FPS com GPU NVIDIA Tesla T4.
- YOLOv6-S: 45,0% de AP a 484 FPS.
- YOLOv6-M: 50,0% de AP a 226 FPS.
- YOLOv6-L: 52,8% de AP a 116 FPS.
- YOLOv6-L6: Precisão de última geração em tempo real.

O YOLOv6 também fornece modelos quantizados para diferentes precisões e modelos otimizados para plataformas móveis.

## Exemplos de Uso

Este exemplo fornece exemplos simples de treinamento e inferência do YOLOv6. Para documentação completa sobre esses e outros [modos](../modes/index.md), consulte as páginas de documentação [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) e [Export](../modes/export.md).

!!! Example "Exemplo"

    === "Python"

        Modelos pré-treinados `*.pt` do PyTorch, assim como arquivos de configuração `*.yaml`, podem ser passados à classe `YOLO()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import YOLO

        # Constrói um modelo YOLOv6n do zero
        model = YOLO('yolov6n.yaml')

        # Exibe informações do modelo (opcional)
        model.info()

        # Treina o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Executa inferência com o modelo YOLOv6n na imagem 'bus.jpg'
        results = model('caminho/para/onibus.jpg')
        ```

    === "CLI"

        Comandos da CLI estão disponíveis para executar diretamente os modelos:

        ```bash
        # Constrói um modelo YOLOv6n do zero e o treina no conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Constrói um modelo YOLOv6n do zero e executa inferência na imagem 'bus.jpg'
        yolo predict model=yolov6n.yaml source=caminho/para/onibus.jpg
        ```

## Tarefas e Modos Suportados

A série YOLOv6 oferece uma variedade de modelos, cada um otimizado para [Detecção de Objetos](../tasks/detect.md) de alta performance. Esses modelos atendem a diferentes necessidades computacionais e requisitos de precisão, tornando-os versáteis para uma ampla variedade de aplicações.

| Tipo de Modelo | Pesos Pré-treinados | Tarefas Suportadas                        | Inferência | Validação | Treinamento | Exportação |
|----------------|---------------------|-------------------------------------------|------------|-----------|-------------|------------|
| YOLOv6-N       | `yolov6-n.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv6-S       | `yolov6-s.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv6-M       | `yolov6-m.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv6-L       | `yolov6-l.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv6-L6      | `yolov6-l6.pt`      | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |

Esta tabela fornece uma visão geral detalhada das variantes do modelo YOLOv6, destacando suas capacidades em tarefas de detecção de objetos e sua compatibilidade com vários modos operacionais, como [inferência](../modes/predict.md), [validação](../modes/val.md), [treinamento](../modes/train.md) e [exportação](../modes/export.md). Esse suporte abrangente garante que os usuários possam aproveitar totalmente as capacidades dos modelos YOLOv6 em uma ampla gama de cenários de detecção de objetos.

## Citações e Agradecimentos

Gostaríamos de agradecer aos autores por suas contribuições significativas no campo da detecção de objetos em tempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

    O artigo original do YOLOv6 pode ser encontrado no [arXiv](https://arxiv.org/abs/2301.05586). Os autores disponibilizaram publicamente seu trabalho, e o código pode ser acessado no [GitHub](https://github.com/meituan/YOLOv6). Agradecemos seus esforços em avançar no campo e disponibilizar seu trabalho para a comunidade em geral.
