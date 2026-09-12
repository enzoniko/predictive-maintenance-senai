# 1. Interpretação do problema

## 1.1 O que está sendo pedido, de fato

O enunciado descreve dois papéis simultâneos para quem assina este documento:

1. **Mentor de IA / arquiteto de projeto**: montar a arquitetura completa e o cronograma de execução de um projeto de manutenção preditiva, *caso ele seja contratado*.
2. **Autor de uma prévia técnica**: usar os dados já disponibilizados para produzir um pipeline de ML completo (preparação → modelagem → avaliação → disponibilização) que **convença o diretor da empresa** a contratar o projeto.

Esses dois papéis puxam em direções diferentes se não forem conciliados. Uma prévia puramente técnica (a melhor métrica possível) não convence sozinha um diretor — ele quer saber se o time entende o problema, se os riscos foram mapeados e se existe um caminho realista até valor de produção. Uma prévia puramente gerencial (só slides e cronograma) não demonstra capacidade de execução. Este projeto foi deliberadamente construído para entregar as duas coisas com o mesmo peso, e a organização do repositório e da documentação reflete essa dualidade: [`README.md`](../README.md) e este diretório `docs/` para a camada de projeto/negócio, `src/pdm/` e `notebooks/` para a camada técnica.

## 1.2 O que o diretor realmente quer ouvir

Um diretor de uma indústria de grande porte que já opera chão de fábrica não precisa ser convencido de que "IA funciona" em abstrato — ele quer saber:

- **Isso funciona nos *meus* dados, com os *meus* sensores?** (viabilidade)
- **O que acontece quando der errado?** (robustez, riscos, plano de mitigação)
- **Quanto tempo e quanto custa até eu ver valor de produção?** (cronograma, marcos)
- **Como eu confio numa decisão automática sobre uma máquina que pode custar caro para parar ou para quebrar?** (interpretabilidade, incerteza, auditabilidade)

Por isso a prévia técnica deste projeto não termina em "F1-macro = 0,96 e pronto". Ela é construída para responder às quatro perguntas acima explicitamente: viabilidade é demonstrada com controles estatísticos (não só uma métrica), robustez é medida com testes de degradação, cronograma e custo têm um capítulo próprio ([`04_cronograma_9_meses.md`](04_cronograma_9_meses.md)), e confiança é endereçada com conjuntos de predição conformais e explicabilidade (ver [`03_arquitetura.md`](03_arquitetura.md) e `src/pdm/evaluation/`).

## 1.3 Ceticismo como método, não como obstáculo

O enunciado contém uma pista deliberada: *"sensores [...] podem captar sinais de ruídos e até mesmo entrarem em falha."* Isso não é um aviso genérico — é um convite para tratar os dados como evidência a ser examinada, não como verdade a ser consumida. A primeira entrega técnica deste projeto não é um modelo, é uma **auditoria de dados** (`src/pdm/data/audit.py`, executável via `python -m pdm.cli audit`), que reavalia cada afirmação do enunciado em vez de assumi-la.

O que essa auditoria encontrou, nos dados reais fornecidos:

| Afirmação do enunciado / suposição razoável | O que a auditoria mostrou |
|---|---|
| "Seis arquivos .npy" | São `Classes.npy` + 5 arquivos `Dados_N.npy` — ou seja, 5 sensores, não 6. Ver [`07_perguntas_ao_cliente.md`](07_perguntas_ao_cliente.md). |
| Cada arquivo tem 200 amostras (20 ms a 10 kHz) | `Dados_1`, `Dados_2` e `Dados_3` têm **201 colunas**. A 201ª é uma coluna quase inteiramente vazia (NaN em 49.999 de 50.000 linhas, com um único valor não-NaN por arquivo, sem relação consistente com a classe) — um artefato de exportação, não uma amostra real. O `loader` (`src/pdm/data/loader.py`) descarta essa coluna, mas só depois de validar que ela realmente se encaixa nesse padrão — um arquivo diferente, com dados de verdade nessa posição, faria a validação falhar alto em vez de perder dados silenciosamente. |
| "Dados adquiridos de forma simultânea" | Não há timestamp nos arquivos — essa afirmação **não pode ser verificada diretamente**. A auditoria usa um proxy indireto (correlação entre a envoltória RMS dos 3 sensores reais linha a linha) que é compatível com aquisição simultânea, mas não a prova. Ver a seção de simultaneidade em `audit.py` e a pergunta correspondente em `07_perguntas_ao_cliente.md`. |
| "Classes podem ou não representar falhas" | As 5 classes (`Classe A`–`E`) estão **perfeitamente balanceadas** (10.000 janelas cada) — um padrão que não ocorre naturalmente numa planta industrial, onde operação saudável domina. Isso é tratado como indício de que o conjunto foi curado/amostrado deliberadamente para fins do case, e a avaliação inclui uma reamostragem sob um prior realista (`src/pdm/evaluation/imbalance.py`) para não superestimar o desempenho esperado em produção. |
| Todos os 5 sensores carregam informação útil | `Dados_4` está travado num valor quase constante (sensor morto/mal cabeado) e `Dados_5` é estatisticamente indistinguível de ruído branco em relação às classes (um classificador treinado só nele fica no nível do acaso). Ambos são **excluídos** do pipeline de features com justificativa estatística (teste de Kruskal-Wallis + teste de permutação de informação mútua), não por inspeção visual. `Dados_1`, `Dados_2` e `Dados_3` carregam sinal real e são os sensores usados no modelo. |

Essa tabela é, em si, um argumento de senioridade: mostra que a primeira pergunta feita aos dados não foi "que modelo eu uso?", mas "o que esses dados realmente contêm, e o que eu não posso saber sem falar com o time de automação do cliente?".

## 1.4 O que a "máquina com motor elétrico" provavelmente é

O enunciado não identifica o tipo de máquina, o tipo de sensor, nem a grandeza física medida (o que é, por si, uma lacuna de requisitos — ver `07_perguntas_ao_cliente.md`). A partir do conteúdo espectral dos 3 sensores retidos (concentração de energia em bandas específicas entre 400 Hz e 1,7 kHz, variável por classe, com saturação por volta de ±5 V — condizente com o fundo de escala típico de um condicionador de sinal), a hipótese de trabalho adotada é que `Dados_1–3` são **canais de vibração ou corrente de um motor elétrico girante** (o enunciado confirma isso), possivelmente 2–3 eixos de um mesmo acelerômetro triaxial ou 3 fases de corrente. Essa hipótese orienta as features de engenharia (domínio do tempo, frequência e wavelet, todas usadas de forma consagrada em diagnóstico de máquinas rotativas) e as bandas de frequência analisadas, mas **é uma hipótese, não um fato confirmado** — o projeto contratado precisa da confirmação do time de automação antes de qualquer feature ser tratada como fisicamente interpretável em unidades de engenharia (ver `src/pdm/preprocessing/dsp.py::UnitConverter` e `configs/sensors.yaml`, que documentam essa lacuna em vez de assumir uma unidade arbitrária).

Uma limitação estrutural importante, e que molda todo o cronograma da fase de pesquisa: a 10 kHz, uma janela de 200 amostras cobre apenas **20 ms**, com resolução espectral de **50 Hz**. Isso é suficiente para capturar energia de banda larga (o que já permite classificar os estados operacionais nos dados fornecidos, como mostrado nos resultados), mas é curto demais para separar frequências características de falhas de rolamento (BPFO/BPFI/BSF/FTF) e suas bandas laterais, que tipicamente exigem resolução sub-Hz e um sinal de rotação (RPM) para *order tracking* — nenhum dos dois está disponível aqui. Isso é registrado como requisito de dados para a fase contratada (ver `02_engenharia_requisitos.md`), não escondido.

## 1.5 O resultado da prévia, em uma frase

Com os 3 sensores retidos pela auditoria, um pipeline de features interpretáveis (tempo + frequência + wavelet) e um classificador não-caixa-preta comparado por validação cruzada entre árvore de decisão, Random Forest, HistGradientBoosting e XGBoost, o modelo final (HistGradientBoosting venceu a comparação) atinge **F1-macro = 0,961** em teste totalmente não visto durante o treino, com erro de calibração esperado (ECE) de apenas **0,005** — as probabilidades que o modelo reporta batem com a frequência real de acerto, o que é o que torna a predição conformal (`evaluation/conformal.py`) confiável em vez de decorativa. Os dois controles de sanidade confirmam que isso é sinal real, não vazamento: um modelo treinado com os rótulos embaralhados cai para F1 = 0,198 (o acaso teórico para 5 classes é 0,20), e um modelo treinado *apenas* no sensor de ruído excluído pela auditoria (`Dados_5`) fica em F1 = 0,067 — abaixo até do acaso, reforçando que essa exclusão foi a decisão certa. Os números completos estão em `models/artifacts/model_bundle.joblib` e são reproduzidos nos slides executivos e técnicos. O que se está vendendo ao diretor não é esse número isolado, mas o **método** por trás dele: um pipeline que desconfia dos dados antes de aprender com eles, que sabe quando não sabe (o conjunto de predição conformal cobre a classe verdadeira em 89% dos casos de teste, muito próximo do alvo de 90% configurado, com tamanho médio de conjunto de 0,98 — ou seja, quase sempre uma única classe confiante), e que já está desenhado para o volume e a variabilidade de uma planta real, não só para este conjunto curado.
