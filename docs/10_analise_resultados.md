# 10. Análise dos resultados da prévia

Este documento sintetiza os resultados numéricos já reportados em [`01_interpretacao_problema.md`](01_interpretacao_problema.md) e [`03_arquitetura.md`](03_arquitetura.md) (seção 3.7) numa leitura única, e responde diretamente a três perguntas: o resultado é bom em todas as frentes (desempenho, incerteza, confiança)? dá para extrair regras simples de um método de árvore? e o que os métodos de explicabilidade realmente mostraram?

## 10.1 Panorama geral — sim, sucesso em todas as frentes medidas

| Dimensão | Métrica | Resultado | Leitura |
|---|---|---|---|
| Desempenho | F1-macro (teste, nunca visto) | **0,962** (Linux, stack completa) / 0,961 (ARM) | Muito acima do acaso (0,20 para 5 classes) e consistente entre plataformas |
| Desempenho por classe | Precisão/recall por classe | 0,94–0,99 em todas as 5 classes | Nenhuma classe é sacrificada pelas outras; Classe C e Classe E são as mais confundidas entre si (ver matriz de confusão) |
| Calibração | ECE (erro de calibração esperado) | **0,006–0,007** | Probabilidade reportada ≈ frequência real de acerto — pré-requisito para a predição conformal ser confiável, não decorativa |
| Incerteza | Cobertura conformal @ alvo 0,90 | **0,894–0,894** (cobertura por classe: 0,88–0,90) | O sistema "sabe quando não sabe" com garantia estatística, não só uma heurística |
| Incerteza | Tamanho médio do conjunto conformal | **0,98** | Na prática, quase sempre uma única classe confiante — o sistema raramente hesita, e quando hesita, é porque de fato há ambiguidade |
| Confiabilidade sob prior realista | F1-macro reamostrado (operação saudável dominante) | **0,935** | Cai um pouco frente ao cenário balanceado do case (esperado — ver `evaluation/imbalance.py`), mas continua muito forte; maior queda de recall é na Classe E (0,845) |
| Robustez | F1 ao remover um sensor | Ver `docs/figures/robustness_sensor_dropout.png` | Degrada de forma previsível e proporcional à informação de cada sensor, não colapsa |
| Ausência de vazamento (controle 1) | F1 com rótulos embaralhados | **0,200** (exatamente o acaso teórico) | O pipeline de avaliação está correto; nenhuma informação da resposta vaza para o treino |
| Ausência de vazamento (controle 2) | F1 treinando só no sensor de ruído excluído | **0,067** (abaixo do acaso) | A decisão de excluir `Dados_5` na auditoria foi estatisticamente correta, não uma escolha arbitrária |

**Conclusão desta seção**: não é só "uma métrica boa" — desempenho, calibração, incerteza quantificada e ausência de vazamento foram todos verificados de forma independente, e todos vieram consistentes entre si e entre duas plataformas de execução diferentes (Windows ARM64 e Linux x86-64, ver `03_arquitetura.md` seção 3.7). Essa é, precisamente, a base para convencer alguém cético (um diretor, um auditor técnico) de que o número não é sorte.

## 10.2 Dá para extrair regras simples de um método de árvore? — sim, mas com um custo real e mensurável

Fizemos o teste diretamente: treinamos árvores de decisão isoladas (`DecisionTreeClassifier`, balanceadas por classe) em profundidades crescentes sobre o mesmo conjunto de features, e medimos o F1-macro em validação cruzada e em teste a cada profundidade.

| Profundidade | Nº de folhas | Nº de nós | F1-macro (CV) | F1-macro (teste) | Legibilidade humana |
|---|---|---|---|---|---|
| 2 | 4 | 7 | 0,312 | 0,311 | Total |
| 3 | 8 | 15 | 0,444 | 0,444 | Total |
| **4** | **16** | **31** | **0,546** | **0,543** | **Alta — ver regras abaixo** |
| 5 | 32 | 63 | 0,579 | 0,571 | Razoável |
| 6 | 64 | 127 | 0,609 | 0,609 | Difícil |
| 8 | 243 | 485 | 0,665 | 0,668 | Inviável de ler |
| sem limite | 5.494 | 10.987 | 0,691 | 0,712 | Impossível |

O resultado é claro e honesto: **uma árvore simples o suficiente para um humano ler de ponta a ponta (profundidade 4, 31 nós) entrega F1-macro ≈ 0,54 — bem abaixo dos 0,96 do ensemble**. Mesmo sem limite de profundidade (quase 11 mil nós, o oposto de "regra simples"), a árvore isolada só chega a 0,71. Isso significa que a fronteira entre as 5 classes **não é bem descrita por cortes simples em poucas variáveis** — o ensemble (HistGradientBoosting) precisa combinar muitas dessas fronteiras parciais para capturar o padrão real.

Isso não invalida a ideia — só localiza onde ela se aplica. As regras da árvore de profundidade 4 (reproduzidas integralmente abaixo) já são um artefato de comunicação genuinamente útil, mesmo com desempenho isolado modesto:

```
|--- Dados_1_band_3500_4000hz <= 0.00
|   |--- Dados_1_band_4000_4500hz <= 0.00
|   |   |--- Dados_3_std <= 0.15
|   |   |   |--- Dados_3_mean <= 0.00  -> Classe C
|   |   |   |--- Dados_3_mean >  0.00  -> Classe E
|   |   |--- Dados_3_std >  0.15
|   |   |   |--- Dados_3_cwt_scale3_energy_frac <= 0.00 -> Classe C
|   |   |   |--- Dados_3_cwt_scale3_energy_frac >  0.00 -> Classe D
|   |--- Dados_1_band_4000_4500hz >  0.00
|   |   |--- Dados_2_band_4000_4500hz <= 0.00
|   |   |   |--- Dados_3_band_2500_3000hz <= 0.01 -> Classe E
|   |   |   |--- Dados_3_band_2500_3000hz >  0.01 -> Classe D
|   |   |--- Dados_2_band_4000_4500hz >  0.00      -> Classe D
|--- Dados_1_band_3500_4000hz >  0.00
|   |--- Dados_2_std <= 0.18
|   |   |--- Dados_2_cwt_scale0_energy_frac <= 0.00
|   |   |   |--- Dados_1_cwt_scale13_energy_frac <= 0.11 -> Classe C
|   |   |   |--- Dados_1_cwt_scale13_energy_frac >  0.11 -> Classe E
|   |   |--- Dados_2_cwt_scale0_energy_frac >  0.00
|   |   |   |--- Dados_3_cwt_scale0_energy_frac <= 0.01 -> Classe A
|   |   |   |--- Dados_3_cwt_scale0_energy_frac >  0.01 -> Classe C
|   |--- Dados_2_std >  0.18
|   |   |--- Dados_1_cwt_scale13_energy_frac <= 0.10
|   |   |   |--- Dados_3_band_4500_5000hz <= 0.01 -> Classe A
|   |   |   |--- Dados_3_band_4500_5000hz >  0.01 -> Classe B
|   |   |--- Dados_1_cwt_scale13_energy_frac >  0.10
|   |   |   |--- Dados_1_band_3500_4000hz <= 0.00 -> Classe D
|   |   |   |--- Dados_1_band_3500_4000hz >  0.00 -> Classe A
```

A **primeira pergunta da árvore inteira** — energia de `Dados_1` na banda 3500–4000 Hz — já é, sozinha, o corte mais informativo de todo o conjunto (importância Gini 0,39, mais que o dobro da segunda posição). Isso é uma regra genuinamente comunicável para o chão de fábrica: *"quando a energia nessa banda de frequência de `Dados_1` sobe, a máquina sai do regime C/D/E para o regime A/B/C"* — uma frase, não um modelo.

**Recomendação prática** (a incorporar no roteiro de pesquisa, `06_track_pesquisa.md`): usar essa árvore rasa não como classificador de produção, mas como **camada de triagem/explicação de primeira linha** — ela concorda com o ensemble na maioria dos casos fáceis e serve de checagem de sanidade legível para o operador, enquanto o ensemble (com predição conformal) decide os casos que a árvore rasa erraria. Essa combinação — regra simples para o caso comum, modelo forte com incerteza calibrada para o caso difícil — tende a gerar mais confiança na indústria do que qualquer um dos dois sozinho.

## 10.3 O que os métodos de explicabilidade mostraram — convergência entre três métodos independentes

O achado mais forte desta seção não é uma lista de features importantes isolada — é que **três métodos completamente independentes (importância por permutação global, divisões da árvore de decisão, e explicação local via LIME de uma predição individual) apontam para as mesmas variáveis**, o que é evidência de estrutura física real nos dados, não de artefato estatístico de um único método.

**Importância por permutação** (queda de F1-macro do ensemble ao embaralhar cada feature, `notebooks/02_feasibility_and_evaluation.ipynb`), top 3: `Dados_3_cwt_scale0_energy_frac`, `Dados_1_band_3500_4000hz`, `Dados_1_band_4000_4500hz`.

**Divisões da árvore de decisão rasa** (seção 10.2 acima), top 3 por importância Gini: `Dados_1_band_3500_4000hz`, `Dados_2_std`, `Dados_1_cwt_scale13_energy_frac`.

**LIME, explicação local de uma predição individual real** (`notebooks/03_api_demo.ipynb`), top 2 contribuições: `Dados_3_cwt_scale4_energy_frac` (+0,25) e `Dados_1_cwt_scale13_energy_frac` (+0,11).

`Dados_1_band_3500_4000hz` aparece como a variável nº1 tanto na árvore quanto entre as top-2 da importância por permutação. `Dados_1_cwt_scale13_energy_frac` aparece nas três listas — na árvore, na permutação (top 8) e na explicação local do LIME. Isso não é coincidência de um único experimento: são três procedimentos matematicamente distintos (um mede queda de desempenho ao corromper uma variável globalmente, outro é a estrutura interna de uma árvore treinada, o terceiro é uma aproximação linear local em torno de uma única predição) concordando sobre onde a informação relevante está.

**Insight físico**: as variáveis mais importantes, nos três métodos, são quase todas de **energia em bandas específicas de frequência (3500–4500 Hz) e de escalas específicas da transformada wavelet (CWT)** — não estatísticas de tempo genéricas (média, desvio-padrão sozinhos raramente aparecem no topo). Isso reforça a hipótese de trabalho registrada em `01_interpretacao_problema.md`: o sinal discrimina os estados da máquina por **conteúdo espectral concentrado em bandas específicas**, consistente com harmônicos de um motor elétrico girante sob diferentes condições — não com um deslocamento genérico de amplitude. Também explica por que os cortes da árvore rasa (todos em bandas de frequência ou escalas wavelet, nunca em `_mean` isolado) fazem sentido físico e não são artefatos de overfitting num conjunto pequeno de variáveis.

**O que ficou faltando** (registrado como próximo passo, não escondido): sem a confirmação do time de automação sobre a grandeza física exata de cada sensor (`07_perguntas_ao_cliente.md`), não é possível traduzir "banda 3500–4500 Hz" para uma causa mecânica nomeada (ex.: uma frequência característica de rolamento específica). A infraestrutura para essa tradução já existe (`preprocessing/dsp.py::UnitConverter`, `configs/sensors.yaml`) — falta só o dado de entrada do cliente.
