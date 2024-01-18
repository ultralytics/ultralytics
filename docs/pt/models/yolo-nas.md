---
comments: true
description: Explore a documentação detalhada do YOLO-NAS, um modelo superior de detecção de objetos. Saiba mais sobre suas funcionalidades, modelos pré-treinados, uso com a API do Ultralytics Python e muito mais.
keywords: YOLO-NAS, Deci AI, detecção de objetos, aprendizado profundo, busca de arquitetura neural, API do Ultralytics Python, modelo YOLO, modelos pré-treinados, quantização, otimização, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## Visão Geral

Desenvolvido pela Deci AI, o YOLO-NAS é um modelo de detecção de objetos inovador. É o produto da tecnologia avançada de Busca de Arquitetura Neural, meticulosamente projetado para superar as limitações dos modelos YOLO anteriores. Com melhorias significativas no suporte à quantização e compromisso entre precisão e latência, o YOLO-NAS representa um grande avanço na detecção de objetos.

![Exemplo de imagem do modelo](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Visão geral do YOLO-NAS.** O YOLO-NAS utiliza blocos que suportam quantização e quantização seletiva para obter um desempenho ideal. O modelo, quando convertido para sua versão quantizada INT8, apresenta uma queda mínima na precisão, uma melhoria significativa em relação a outros modelos. Esses avanços culminam em uma arquitetura superior com capacidades de detecção de objetos sem precedentes e desempenho excepcional.

### Principais Características

- **Bloco Básico Amigável para Quantização:** O YOLO-NAS introduz um novo bloco básico que é amigo da quantização, abordando uma das limitações significativas dos modelos YOLO anteriores.
- **Treinamento e Quantização Sofisticados:** O YOLO-NAS utiliza esquemas avançados de treinamento e quantização pós-treinamento para melhorar o desempenho.
- **Otimização AutoNAC e Pré-Treinamento:** O YOLO-NAS utiliza a otimização AutoNAC e é pré-treinado em conjuntos de dados proeminentes, como COCO, Objects365 e Roboflow 100. Esse pré-treinamento torna o modelo extremamente adequado para tarefas de detecção de objetos em ambientes de produção.

## Modelos Pré-Treinados

Experimente o poder da detecção de objetos de última geração com os modelos pré-treinados do YOLO-NAS fornecidos pela Ultralytics. Esses modelos foram projetados para oferecer um desempenho excelente em termos de velocidade e precisão. Escolha entre várias opções adaptadas às suas necessidades específicas:

| Modelo           | mAP   | Latência (ms) |
|------------------|-------|---------------|
| YOLO-NAS S       | 47.5  | 3.21          |
| YOLO-NAS M       | 51.55 | 5.85          |
| YOLO-NAS L       | 52.22 | 7.87          |
| YOLO-NAS S INT-8 | 47.03 | 2.36          |
| YOLO-NAS M INT-8 | 51.0  | 3.78          |
| YOLO-NAS L INT-8 | 52.1  | 4.78          |

Cada variante do modelo foi projetada para oferecer um equilíbrio entre Precisão Média Média (mAP) e latência, ajudando você a otimizar suas tarefas de detecção de objetos em termos de desempenho e velocidade.

## Exemplos de Uso

A Ultralytics tornou os modelos YOLO-NAS fáceis de serem integrados em suas aplicações Python por meio de nosso pacote `ultralytics`. O pacote fornece uma API Python de fácil utilização para simplificar o processo.

Os seguintes exemplos mostram como usar os modelos YOLO-NAS com o pacote `ultralytics` para inferência e validação:

### Exemplos de Inferência e Validação

Neste exemplo, validamos o YOLO-NAS-s no conjunto de dados COCO8.

!!! Example "Exemplo"

    Este exemplo fornece um código simples de inferência e validação para o YOLO-NAS. Para lidar com os resultados da inferência, consulte o modo [Predict](../modes/predict.md). Para usar o YOLO-NAS com modos adicionais, consulte [Val](../modes/val.md) e [Export](../modes/export.md). O YOLO-NAS no pacote `ultralytics` não suporta treinamento.

    === "Python"

        Arquivos de modelos pré-treinados `*.pt` do PyTorch podem ser passados para a classe `NAS()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import NAS

        # Carrega um modelo YOLO-NAS-s pré-treinado no COCO
        model = NAS('yolo_nas_s.pt')

        # Exibe informações do modelo (opcional)
        model.info()

        # Valida o modelo no conjunto de dados de exemplo COCO8
        results = model.val(data='coco8.yaml')

        # Executa inferência com o modelo YOLO-NAS-s na imagem 'bus.jpg'
        results = model('caminho/para/bus.jpg')
        ```

    === "CLI"

        Comandos de CLI estão disponíveis para executar diretamente os modelos:

        ```bash
        # Carrega um modelo YOLO-NAS-s pré-treinado no COCO e valida seu desempenho no conjunto de dados de exemplo COCO8
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Carrega um modelo YOLO-NAS-s pré-treinado no COCO e executa inferência na imagem 'bus.jpg'
        yolo predict model=yolo_nas_s.pt source=caminho/para/bus.jpg
        ```

## Tarefas e Modos Compatíveis

Oferecemos três variantes dos modelos YOLO-NAS: Pequeno (s), Médio (m) e Grande (l). Cada variante foi projetada para atender a diferentes necessidades computacionais e de desempenho:

- **YOLO-NAS-s**: Otimizado para ambientes com recursos computacionais limitados, mas eficiência é fundamental.
- **YOLO-NAS-m**: Oferece uma abordagem equilibrada, adequada para detecção de objetos em geral com maior precisão.
- **YOLO-NAS-l**: Adaptado para cenários que requerem a maior precisão, onde os recursos computacionais são menos restritos.

Abaixo está uma visão geral detalhada de cada modelo, incluindo links para seus pesos pré-treinados, as tarefas que eles suportam e sua compatibilidade com diferentes modos de operação.

| Tipo de Modelo | Pesos Pré-Treinados                                                                           | Tarefas Suportadas                        | Inferência | Validação | Treinamento | Exportação |
|----------------|-----------------------------------------------------------------------------------------------|-------------------------------------------|------------|-----------|-------------|------------|
| YOLO-NAS-s     | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_s.pt) | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ❌           | ✅          |
| YOLO-NAS-m     | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_m.pt) | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ❌           | ✅          |
| YOLO-NAS-l     | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_l.pt) | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ❌           | ✅          |

## Citações e Agradecimentos

Se você utilizar o YOLO-NAS em seus estudos ou trabalho de desenvolvimento, por favor, cite o SuperGradients:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

Expressamos nossa gratidão à equipe [SuperGradients](https://github.com/Deci-AI/super-gradients/) da Deci AI por seus esforços na criação e manutenção deste recurso valioso para a comunidade de visão computacional. Acreditamos que o YOLO-NAS, com sua arquitetura inovadora e capacidades superiores de detecção de objetos, se tornará uma ferramenta fundamental para desenvolvedores e pesquisadores.

*keywords: YOLO-NAS, Deci AI, detecção de objetos, aprendizado profundo, busca de arquitetura neural, API do Ultralytics Python, modelo YOLO, SuperGradients, modelos pré-treinados, bloco básico amigável para quantização, esquemas avançados de treinamento, quantização pós-treinamento, otimização AutoNAC, COCO, Objects365, Roboflow 100*
