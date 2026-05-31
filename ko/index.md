---
layout: distill
lang: ko
translation_key: index
permalink: /ko/
title: "모델을 스케일하는 법"
subtitle: "TPU 위 LLM을 바라보는 시스템 관점"
description: "LLM 학습은 종종 연금술처럼 느껴지지만, 그렇다고 모델의 성능을 이해하고 최적화하는 것까지 그래야만 하는 것은 아닙니다. 이 책은 TPU (또는 GPU)가 어떻게 서로 통신하고, LLM들이 어떻게 실제 하드웨어에서 동작하며, 이러한 모델들의 학습 및 추론 단계에서 어떻게 병렬화(parallelize)를 적용하여 초대규모 환경에서 효율적으로 동작하게 하는지를 알아보고, 언어 모델 스케일링의 과학을 풀어내는 것을 목적으로 합니다. 여러분이 \"이 LLM을 학습하려면 얼마나 많은 비용이 필요할까?\", \"이 모델을 직접 서빙하려면 얼마나 많은 메모리가 필요할까?\", 또는 \"AllGather가 도대체 뭐지?\"라는 의문을 가져보신 적이 있다면, 이 책이 도움이 되시기를 바랍니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: ../roofline
next_section_name: "1부: Roofline"

bibliography: main.bib

citation: true

translation:
  date: 2026-05-31
  translators:
    - name: Gio Paik
      url: "https://sites.google.com/view/giopaik"

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

toc:
  - name: 큰 그림
  - name: 각 장 링크

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

딥러닝의 대부분은 여전히 일종의 흑마법처럼 여겨지곤 하지만, 모델의 성능을 최적화하는 것까지 그렇게 여길 필요는 없습니다. (대규모 환경에서도요!) 단 하나의 가속기만 있는 환경에서부터 수천, 수만 개의 가속기가 탑재된 환경까지 모든 곳에서 통하는 비교적 단순한 원칙들이 존재하기 때문입니다. 이 원칙들을 이해하면, 여러분은 다음과 같은 유용한 작업들을 해낼 수 있습니다:

- 모델이 이론상 최대 성능에 얼마나 가까운지 가늠할 수 있습니다.
- 각기 다른 규모에서 어떤 병렬화 기법을 적용할지(여러 디바이스에 컴퓨테이션을 어떻게 분배할지) 충분한 정보에 입각한 결정을 내릴 수 있습니다.
- 대규모 트랜스포머 모델의 학습과 구동에 요구되는 비용과 시간을 추정할 수 있습니다.
- [특정](https://arxiv.org/abs/2205.14135) [하드웨어](https://arxiv.org/abs/1911.02150) [특성](https://arxiv.org/abs/2007.00072)에 맞는 알고리즘을 설계할 수 있습니다.
- 알고리즘 성능을 제한하는 원인에 대한 명확한 이해를 바탕으로 하드웨어를 설계할 수 있습니다.

**읽기 전 필요한 지식:** 이 책은 여러분이 LLM과 트랜스포머 구조에 대한 기본적인 이해를 가지고 있다고 가정합니다. 다만, 이러한 모델들이 어떻게 확장되는지까지 알고 계시진 않아도 좋습니다. LLM 학습에 대한 기본적인 지식, 이상적으로는 JAX에 대해 어느 정도 알고 계신다면 도움이 됩니다. 필요하시다면, 트랜스포머 구조에 대해 다룬 [이 블로그 포스트](https://jalammar.github.io/illustrated-transformer/) 또는 [트랜스포머 논문](https://arxiv.org/abs/1706.03762)을 읽어보세요. 추가로, [다음 리스트](../conclusion#further-reading)에 함께 읽거나, 본 책을 완독하시고 읽기 좋은 자료들이 정리되어 있습니다.

**목표 & 피드백:** 이 책을 모두 읽으셨을 때, 주어진 하드웨어 플랫폼에서 트랜스포머 모델을 위한 적절한 병렬화 기법과 학습 및 추론에 소요되는 시간을 대략적으로 예측할 수 있다면 이 책의 목표를 달성한 것입니다. 만약 이해에 어려움이 있다면, 이메일 또는 댓글로 저희에게 알려주세요! 함께 책을 더 이해하기 쉽게 개선해 봅시다.

<p markdown=1 class="announce">NVIDIA GPU 환경에 대해 다루는 새로운 [Section 12](../gpus)도 읽어보세요!</p>

### 왜 이걸 배워야 하나요?

3, 4년 전까지만 해도, 저 역시 대부분의 ML 연구자들이 이 책의 내용 대부분을 이해할 필요가 없다고 생각했습니다. 그러나 오늘날, "small" 모델들조차도 대부분 하드웨어 성능을 거의 끝까지 사용하게 되었고, 새로운 연구를 위해서는 누구나 각자 가용한 자원에서 효율성을 따질 필요가 있게 되었습니다.<d-footnote>역사적으로, ML 연구는 시스템 혁신과 소프트웨어 개선의 틱톡 사이클을 따라왔습니다. Alex Krizhevsky는 CNN을 가속하기 위한 CUDA 코드를 손수 작성해야 했지만, 몇 년 만에 Theano나 TensorFlow와 같은 라이브러리들이 등장하여 더 이상 모든 사람들이 직접 CUDA 코드를 작성할 필요는 없어졌습니다. 어쩌면 몇 년 안에 이 분야에서도 그런 일이 일어나, 이 책에 있는 모든 내용이 추상화되어 사라질지도 모릅니다. 그러나 스케일링 법칙은 모델들을 끊임없이 하드웨어의 한계까지 밀어붙이고 있으며, 적어도 한동안은 최첨단 연구가 모델을 대규모 하드웨어 토폴로지에 맞게 최적화하는 것과 떼려야 뗄 수 없을 것으로 보입니다. </d-footnote>**벤치마크 성능 20% 개선의 대가가 루프라인 효율 20% 저하라면 의미가 없습니다.** 주기적으로 새롭고 유망한 모델 구조들이 제안되지만, 효율적으로 확장 가능하지 않거나 혹은 아무도 그렇게 만들기 위한 작업을 하지 않아서 실패합니다.

**"모델 스케일링"의 목표는 학습 또는 추론에 사용되는 칩의 수를 늘릴 때, 처리량의 증가를 비례적, 선형적으로 유지하는 것에 있습니다.** 이를 "*strong scaling*"이라고 합니다. 칩을 추가하는 것("병렬화")은 계산 시간을 감소시키지만, 동시에 칩 간의 통신 비용은 증가하게 됩니다. 어느 순간 통신에 소요되는 시간이 계산 시간보다 길어지면, 이를 "communication bound"라고 하며, 더 이상 strong scaling이 불가능하게 됩니다. <d-footnote>계산 시간이 감소함에 따라, 단일 칩 수준에서도 병목을 마주할 수 있습니다. 여러분의 멋진 새 TPU 또는 GPU가 초당 500조(T) 회의 연산을 수행할 수 있다고 하더라도, 최적화가 부족할 경우 칩은 메모리에서 파라미터를 옮기는 걸 기다리느라 10배는 더 느리게 동작할 수도 있습니다. 칩당 연산(per-chip computation), 메모리 대역폭, 그리고 전체 메모리 용량 간의 상호작용이 스케일링 이야기에서 핵심적인 이유입니다.</d-footnote> 이러한 병목들이 어디서 발생할지 예측할 수 있다면, 우리는 모델이 병목을 피할 수 있도록 설계하거나, 재구성할 수 있습니다.<d-footnote>하드웨어 설계자들은 반대의 문제를 풀고 있습니다: 최소한의 비용으로 우리의 알고리즘을 돌리기에 충분한 계산량, 대역폭, 메모리 용량을 가진 하드웨어를 만들어야 하는 문제죠. 이 "협력 설계" 문제가 얼마나 까다로운지 상상이 되실 겁니다. 하드웨어 생산자는 이 제품이 실제로 막 선보여질 시기(일반적으로 2, 3년 뒤)에 어떤 알고리즘이 구동될지 예상해야 합니다. 이 게임에서 TPU는 엄청난 성공을 거두었습니다. 행렬곱은 특이할 정도로 대부분의 다른 알고리즘들보다 메모리 바이트당 연산량(N FLOPs per byte)이 높은 알고리즘으로, 시스톨릭 어레이 구조 기반의 초기 TPU는 동시기의 GPU들에 비해 뛰어난 가성비를 보일 수 있었습니다. TPU는 ML 워크로드를 위해 설계되었고, 텐서 코어를 탑재한 GPU들 역시 이러한 틈새시장을 빠르게 공략해 나가고 있습니다. 그러나 만약 신경망 기법들이 지금처럼 성공하지 않았거나, (근본적으로 GPU보다 유연성이 떨어지는) TPU가 처리할 수 없는 방식으로 발전했다면 TPU가 얼마나 큰 손실을 야기했을지 상상해보세요.</d-footnote>

*이 책의 목표는 TPU와 GPU 하드웨어가 어떻게 동작하고, 트랜스포머 구조가 이러한 하드웨어에서 잘 동작하기 위해 어떻게 발전해 왔나 설명하는 것입니다. 이 책이 새로운 아키텍처를 디자인하는 연구자들과 현세대의 LLM을 가속화하고자 하는 엔지니어들 모두에게 도움이 되기를 바랍니다.*

## 큰 그림

이 책은 대략적으로 다음과 같이 구성되어 있습니다:

[1장](../roofline)에서는 루프라인 분석과 우리의 스케일링을 제한할 수 있는 제약요소들(통신, 계산, 메모리)을 알아봅니다. [2장](../tpus)과 [3장](../sharding)에서는 TPU가 어떻게 동작하는지, 특히 여러 칩이 제한적인 대역폭과 지연시간을 갖고 상호 연결된 환경에서 어떻게 동작하는지 살펴봅니다. 구체적으로, 다음과 같은 질문에 대한 답을 다뤄볼 것입니다:

* 특정 크기의 행렬곱 연산에는 얼마만큼의 시간이 필요할까요? 그리고 이건 연산, 메모리, 통신 대역폭 중 어떤 요소에 의해 제약될까요?
* 학습 클러스터를 구축하기 위해, TPU들은 어떤 식으로 연결될까요? 시스템의 각 파트는 얼마만큼의 대역폭을 가질까요?
* 여러 TPU 사이에서 어레이를 취합, 분산, 재분배하는 작업에는 얼마만큼의 시간이 필요할까요?
* 여러 디바이스에 분산되어있는 행렬을 효율적으로 곱하려면 어떻게 해야할까요?

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>그림:</b> elementwise 연산을 수행하는 TPU를 나타낸 <a href='../tpus'>2장</a>에 등장하는 그림. 어레이의 크기와 다양한 연결들의 대역폭에 따라, 우리는 연산이 compute-bound (하드웨어의 최대 연산 능력을 사용하고 있는 상태)인지 또는 memory-bound (메모리 로딩 속도에 의한 병목이 발생하는 상태)인지 판단할 수 있습니다." %}

5년 전만 하더라도 ML 현장에서는 ConvNets, LSTMs, MLPs 등 다양한 아키텍처가 탐구되고 활용되었으나, 최근에는 거의 모든 경우에 트랜스포머<d-cite key="transformers"></d-cite>가 사용되고 있는 상황입니다. 우리는 트랜스포머 구조의 모든 요소들(모든 행렬들의 정확한 크기, normalization이 일어나는 장소, 각 파트에 얼마나 많은 파라미터가 사용되고, 얼마나 많은 FLOPs<d-footnote>부동소수점 연산(FLoating point OPs), 쉽게 말해 덧셈과 곱셈 연산의 개수를 의미합니다. 많은 자료에서 FLOPs를 "초당 연산량"을 나타내기 위해 사용하지만, 우리는 명시적으로 FLOPs/s로 표기하겠습니다.</d-footnote>가 투입되는지)을 이해할 가치가 있다고 확신합니다. [4장](../transformers)에서는 이러한 "트랜스포머의 수학"을 다루며, 학습과 추론 단계에서 파라미터의 개수와 FLOPs를 계산하는 방법을 배워보겠습니다. 이를 통해, 모델이 얼마나 많은 메모리를 사용할지, 연산과 통신에 얼마나 많은 시간을 사용할지, feed-forward 블록보다 attention이 중요해지는 타이밍은 언제일지 알 수 있습니다.

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>그림:</b> 행렬곱(matmul) 연산을 원 안의 점 기호로 표기하여 나타낸 표준적인 트랜스포머 계층. (Norms를 제외한) 모든 파라미터는 보라색으로 표기. <a href='../transformers'>4장</a>에서 이 그림에 대해 깊게 다룰 것입니다." %}

[5장: Training](../training)과 [7장: Inference](../inference)은 이 책의 핵심 파트입니다. 여기서 우리는 "특정 크기의 모델과 특정 개수의 칩이 주어졌을 때, 어떻게 모델의 strong scaling 상태를 유지하면서 병렬화를 적용할 것인가?"라는 질문에 대해 논의합니다. 이 단순한 질문에 대한 답은 놀랍도록 복잡합니다. 크게 보면, 모델을 여러 개의 칩으로 나누기 위한 4개의 주요 병렬화 기법(**data**, **tensor**, **pipeline**, **expert**)과 메모리 사용량을 절감하기 위한 몇 가지 기법(**rematerialization**, **optimizer/model sharding (aka ZeRO)**, **host offload**, **gradient accumulation**)이 있습니다. 우리는 5, 7장에서 이러한 기법들에 대해 설명할 것입니다.

여러분이 이러한 과정을 통해, 새로운 구조와 환경에서 스스로 적절한 기법을 선택할 수 있기를 바랍니다. [6장](../applied-training)과 [8장](../applied-inference)은 이러한 개념들을 유명한 오픈소스 모델인 LLaMA 3에 적용해보는 실습 튜토리얼로 구성되어 있습니다.

마지막으로, [9장](../profiling)과 [10장](../jax-stuff)에서는 이러한 개념들을 어떻게 JAX로 구현하고, 문제가 생겼을 때 어떻게 프로파일링 및 디버깅을 수행하는지 살펴보겠습니다. [12장](../gpus)은 GPU에 대해 다루는 새로운 섹션입니다.

이 자료는 여러분이 스스로 풀어볼 수 있는 문제들을 제공하려고 노력했습니다. 모든 부분을 다 읽거나 순서대로 읽어야 한다는 부담감은 가지지 않으셔도 됩니다. 그리고 피드백을 남겨주시면 감사하겠습니다. 현재는 초안 단계이며, 앞으로도 계속 수정될 예정입니다. 감사합니다!

*이 책에 담긴 많은 아이디어의 원천이 되어준 James Bradbury와 Blake Hechtman에게 감사를 전합니다.*

<h3 markdown=1 class="next-section">그럼 바로, [1장](../roofline)에서 TPU 루프라인에 대해 설명드리겠습니다.</h3>

## 각 섹션 링크

*이 시리즈는 생각보다 방대한 내용으로 구성되어 있지만, 두려워하실 필요는 없습니다. 앞부분 3개의 챕터는 주로 사전지식에 대해 다루며, 여러분이 이미 해당 내용들에 대해 익숙하시다면 건너뛰어도 괜찮습니다. 뒷부분 3개 챕터는 실제 모델들에서의 작업을 다루는만큼, 아마 가장 실용적인 부분일 것입니다.*

**파트 1: 사전지식**

* [**챕터 1: 루프라인 분석에 대한 간단한 소개**](../roofline). 알고리즘은 연산, 통신, 메모리 3개의 요소에 의해 바운드(제약)됩니다. 우리는 이를 통해, 알고리즘이 얼마나 빠를지 가늠해볼 수 있습니다.

* [**챕터 2: TPU 이해하기**](../tpus). TPU는 어떻게 동작할까요? 이게 우리가 학습하고 서빙 가능한 모델에는 어떤 영향을 미칠까요?

* [**챕터 3: 행렬의 샤딩과 곱하기**](../sharding). 모델을 샤딩하고, 여러 TPU에서 병렬화하는 방법을 (샤딩된) 행렬의 곱셈 연산을 통해 알아봅니다.

**파트 2: 트랜스포머**

* [**챕터 4: 당신이 알아야 하는 트랜스포머의 수학**](../transformers). 트랜스포머의 forward, backward pass에는 얼마나 많은 FLOPs가 존재할까요? 파라미터의 개수를 세려면 어떻게 하나요? KV 캐시의 크기는 얼마일까요? 이와 같은 트랜스포머의 수학을 다룹니다.

* [**챕터 5: 트랜스포머 학습을 위한 병렬화 기법들**](../training). FSDP. Megatron sharding. Pipeline parallelism. 몇 개의 칩이 주어질 때, 특정 배치 크기에서 효율적으로 모델을 학습하려면 어떻게 해야 할까요?

* [**챕터 6: TPU에서 LLaMA 3 학습하기**](../applied-training). LLaMA 3를 TPU에서 학습하려면 어떻게 해야하나요? 얼마나 많은 시간이 필요하고, 얼마나 많은 비용이 발생할까요?

* [**챕터 7: 트랜스포머 추론의 모든 것**](../inference). 모델을 학습했다면, 서빙을 해야겠죠. 추론 단계에서는 지연시간이라는 새로운 고려사항이 추가되고, 메모리 사용 구조도 달라집니다. 분산형 서빙과 KV 캐시에 대한 내용들을 배워봅시다.

* [**챕터 8: TPU에서 LLaMA 3 서빙하기**](../applied-inference). TPU v5e에서 LLaMA 3를 서빙하려면 비용이 얼마나 발생할까요? Latency/throughput 트레이드오프가 뭘까요?

**파트 3: 실전 튜토리얼**

* [**챕터 9: TPU Code 프로파일링하기**](../profiling). 실제 LLM은 절대 위에서 설명한 이론처럼 단순하지 않습니다. 여기서 우리는 JAX와 XLA 스택을 배우고, JAX/TensorBoard 프로파일러를 통한 디버깅을 통해 실제 문제를 해결하는 법을 배웁니다.

* [**챕터 10: JAX로 TPU 프로그래밍하기**](../jax-stuff). JAX는 병렬 컴퓨팅을 위한 다양하고 편리한 API들을 제공합니다. 재밌는 예시들과 풀이과정을 통해 배워봅시다.

**파트 4: 결론 및 보너스 콘텐츠**

* [**챕터 11: 결론 및 추가 자료**](../conclusion). 결론과 TPU와 LLM 관련 추가로 학습하면 좋은 자료들.

* [**챕터 12: GPU**](../gpus). GPU에 대한 보너스 섹션. GPU가 어떻게 동작하고, 어떻게 통신하는지, 그리고 GPU의 루프라인이 TPU와 어떻게 다른지 알아봅니다.
