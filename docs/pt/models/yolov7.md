---
comments: true
description: Explore o YOLOv7, um detector de objetos em tempo real. Entenda sua velocidade superior, impressionante precisão e foco exclusivo em otimização treinável de recursos gratuitos.
keywords: YOLOv7, detector de objetos em tempo real, state-of-the-art, Ultralytics, conjunto de dados MS COCO, reparametrização de modelo, atribuição dinâmica de rótulo, escalonamento estendido, escalonamento composto
---

# YOLOv7: Treinável Bag-of-Freebies

O YOLOv7 é um detector de objetos em tempo real state-of-the-art que supera todos os detectores de objetos conhecidos em termos de velocidade e precisão na faixa de 5 FPS a 160 FPS. Ele possui a maior precisão (56,8% de AP) entre todos os detectores de objetos em tempo real conhecidos com 30 FPS ou mais no GPU V100. Além disso, o YOLOv7 supera outros detectores de objetos, como YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5 e muitos outros em velocidade e precisão. O modelo é treinado no conjunto de dados MS COCO do zero, sem usar outros conjuntos de dados ou pesos pré-treinados. O código-fonte para o YOLOv7 está disponível no GitHub.

![Comparação YOLOv7 com outros detectores de objetos](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**Comparação de detectores de objetos state-of-the-art.
** A partir dos resultados na Tabela 2, sabemos que o método proposto tem a melhor relação velocidade-precisão de forma abrangente. Se compararmos o YOLOv7-tiny-SiLU com o YOLOv5-N (r6.1), nosso método é 127 FPS mais rápido e 10,7% mais preciso em AP. Além disso, o YOLOv7 tem 51,4% de AP em uma taxa de quadros de 161 FPS, enquanto o PPYOLOE-L com o mesmo AP tem apenas uma taxa de quadros de 78 FPS. Em termos de uso de parâmetros, o YOLOv7 é 41% menor do que o PPYOLOE-L. Se compararmos o YOLOv7-X com uma velocidade de inferência de 114 FPS com o YOLOv5-L (r6.1) com uma velocidade de inferência de 99 FPS, o YOLOv7-X pode melhorar o AP em 3,9%. Se o YOLOv7-X for comparado com o YOLOv5-X (r6.1) de escala similar, a velocidade de inferência do YOLOv7-X é 31 FPS mais rápida. Além disso, em termos da quantidade de parâmetros e cálculos, o YOLOv7-X reduz 22% dos parâmetros e 8% dos cálculos em comparação com o YOLOv5-X (r6.1), mas melhora o AP em 2,2% ([Fonte](https://arxiv.org/pdf/2207.02696.pdf)).

## Visão Geral

A detecção de objetos em tempo real é um componente importante em muitos sistemas de visão computacional, incluindo rastreamento de múltiplos objetos, direção autônoma, robótica e análise de imagens médicas. Nos últimos anos, o desenvolvimento de detecção de objetos em tempo real tem se concentrado em projetar arquiteturas eficientes e melhorar a velocidade de inferência de várias CPUs, GPUs e unidades de processamento neural (NPUs). O YOLOv7 suporta tanto GPUs móveis quanto dispositivos GPU, desde a borda até a nuvem.

Ao contrário dos detectores de objetos em tempo real tradicionais que se concentram na otimização de arquitetura, o YOLOv7 introduz um foco na otimização do processo de treinamento. Isso inclui módulos e métodos de otimização projetados para melhorar a precisão da detecção de objetos sem aumentar o custo de inferência, um conceito conhecido como "treinável bag-of-freebies".

## Recursos Principais

O YOLOv7 apresenta vários recursos principais:

1. **Reparametrização do Modelo**: O YOLOv7 propõe um modelo reparametrizado planejado, que é uma estratégia aplicável a camadas em diferentes redes com o conceito de caminho de propagação de gradiente.

2. **Atribuição Dinâmica de Rótulo**: O treinamento do modelo com várias camadas de saída apresenta um novo problema: "Como atribuir alvos dinâmicos para as saídas de diferentes ramificações?" Para resolver esse problema, o YOLOv7 introduz um novo método de atribuição de rótulo chamado atribuição de rótulo orientada por liderança de granularidade fina (coarse-to-fine).

3. **Escalonamento Estendido e Composto**: O YOLOv7 propõe métodos de "escalonamento estendido" e "escalonamento composto" para o detector de objetos em tempo real que podem utilizar efetivamente parâmetros e cálculos.

4. **Eficiência**: O método proposto pelo YOLOv7 pode reduzir efetivamente cerca de 40% dos parâmetros e 50% dos cálculos do detector de objetos em tempo real state-of-the-art, além de apresentar uma velocidade de inferência mais rápida e maior precisão de detecção.

## Exemplos de Uso

No momento em que este texto foi escrito, a Ultralytics ainda não oferece suporte aos modelos YOLOv7. Portanto, qualquer usuário interessado em usar o YOLOv7 precisará se referir diretamente ao repositório do YOLOv7 no GitHub para obter instruções de instalação e uso.

Aqui está uma breve visão geral das etapas típicas que você pode seguir para usar o YOLOv7:

1. Acesse o repositório do YOLOv7 no GitHub: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. Siga as instruções fornecidas no arquivo README para a instalação. Isso normalmente envolve clonar o repositório, instalar as dependências necessárias e configurar quaisquer variáveis de ambiente necessárias.

3. Após a conclusão da instalação, você pode treinar e usar o modelo conforme as instruções de uso fornecidas no repositório. Isso geralmente envolve a preparação do conjunto de dados, a configuração dos parâmetros do modelo, o treinamento do modelo e, em seguida, o uso do modelo treinado para realizar a detecção de objetos.

Observe que as etapas específicas podem variar dependendo do caso de uso específico e do estado atual do repositório do YOLOv7. Portanto, é altamente recomendável consultar diretamente as instruções fornecidas no repositório do YOLOv7 no GitHub.

Lamentamos qualquer inconveniente que isso possa causar e nos esforçaremos para atualizar este documento com exemplos de uso para a Ultralytics assim que o suporte para o YOLOv7 for implementado.

## Citações e Agradecimentos

Gostaríamos de agradecer aos autores do YOLOv7 por suas contribuições significativas no campo da detecção de objetos em tempo real:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

O artigo original do YOLOv7 pode ser encontrado no [arXiv](https://arxiv.org/pdf/2207.02696.pdf). Os autores disponibilizaram publicamente seu trabalho, e o código pode ser acessado no [GitHub](https://github.com/WongKinYiu/yolov7). Agradecemos seus esforços em avançar o campo e tornar seu trabalho acessível à comunidade em geral.
