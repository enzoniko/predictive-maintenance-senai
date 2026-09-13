# 2. Engenharia de requisitos

O enunciado lista "Engenharia de Requisitos" como uma etapa do pipeline. Interpretamos isso em dois sentidos que se complementam: (a) os requisitos de **engenharia de features** do próprio pipeline de ML (cobertos em `src/pdm/features/` e discutidos em [`01_interpretacao_problema.md`](01_interpretacao_problema.md) e [`08_revisao_literatura.md`](08_revisao_literatura.md)); e (b) os **requisitos de projeto de software** no sentido clássico (funcionais, não-funcionais, de dados e de integração) que qualquer projeto contratado precisa ter explícitos antes da fase de execução. Este documento cobre o segundo sentido.

## 2.1 Requisitos funcionais

| ID | Requisito | Onde é atendido nesta prévia |
|---|---|---|
| RF-01 | O sistema deve classificar o estado operacional da máquina a partir de uma janela de sinais multi-sensor. | `src/pdm/models/`, `POST /predict` (`src/pdm/serving/api.py`) |
| RF-02 | O sistema deve indicar quando não tem confiança suficiente para decidir, em vez de forçar uma classe. | Conjuntos de predição conformal (`src/pdm/evaluation/conformal.py`), campo `conformal_set` e `warnings` na resposta da API |
| RF-03 | O sistema deve explicar, por predição, quais variáveis mais pesaram na decisão. | `POST /explain` (SHAP quando disponível, importância de features como alternativa) |
| RF-04 | O sistema deve auditar a qualidade dos dados de entrada antes de confiar neles. | `GET /audit`, `src/pdm/data/audit.py` |
| RF-05 | O sistema deve registrar toda predição, auditoria e verificação de *drift* de forma persistente e consultável. | `src/pdm/serving/db.py` (SQLite local / Postgres em produção) |
| RF-06 | O sistema deve detectar quando a distribuição dos dados de produção se afasta da distribuição de treino. | `GET /drift`, `src/pdm/evaluation/drift.py` |
| RF-07 (evolução futura, fora do escopo contratado inicial) | O sistema deve suportar detecção de anomalias/estados nunca vistos no treino (*open-set*), não só classificação fechada entre as classes conhecidas. | Não implementado na prévia; candidato da trilha de pesquisa interna do SENAI (H4/H8), promovido a produção só se aprovado no critério de `06_track_pesquisa.md` |

## 2.2 Requisitos não-funcionais

- **RNF-01 Portabilidade**: o sistema deve rodar de forma reprodutível no ambiente de implantação alvo (Linux x86-64), sem depender de configuração manual (ver `03_arquitetura.md`, seção 3.6): toda dependência opcional ausente tem fallback documentado no código, nunca falha silenciosamente.
- **RNF-02 Reprodutibilidade**: toda métrica reportada deve ser reproduzível a partir de uma seed fixa (`configs/default.yaml: random_seed`) e do mesmo conjunto de dados versionado.
- **RNF-03 Auditabilidade**: nenhuma decisão de exclusão de sensor ou feature deve ser feita "no olho" (precisa de um teste estatístico documentado (ver `data/audit.py`)).
- **RNF-04 Latência de inferência**: uma predição individual deve responder em menos de 1 s no hardware alvo (não medido formalmente na prévia; folga grande dado que o modelo final é uma árvore/ensemble raso, não uma rede profunda).
- **RNF-05 Sem downtime para retrain**: o retrain de um modelo não pode exigir parar o serviço de inferência (ver `09_mlops.md`, estratégia blue/green de modelo).
- **RNF-06 Custo de storage controlado**: o banco de dados deste projeto armazena predições/auditorias/relatórios de *drift*, não réplicas do histórico bruto de sensores (que já vive no banco do time de automação/software do cliente (ver 2.4)).

## 2.3 Requisitos de dados (o que falta para sair da prévia e ir para produção)

Estes são, na prática, um subconjunto direto das perguntas em [`07_perguntas_ao_cliente.md`](07_perguntas_ao_cliente.md), reformuladas como requisitos formais de entrada do projeto contratado:

- **RD-01**: identificação da grandeza física de cada sensor retido (vibração/aceleração, corrente, tensão, etc.), sensibilidade do transdutor e ganho do condicionador (necessário para converter as features de volts para unidades de engenharia (`src/pdm/preprocessing/dsp.py::UnitConverter`), hoje operando em modo identidade por falta dessa informação).
- **RD-02**: semântica de negócio de cada classe (`Classe A`–`E`): qual é "operação normal", quais são falhas, e qual a severidade/urgência de cada uma. Sem isso, o sistema classifica *padrões*, não *falhas*, e não há como priorizar alarmes.
- **RD-03**: taxa de amostragem efetiva e tamanho de janela maiores (ou um *stream* contínuo com timestamp) para viabilizar a fase de pesquisa em diagnóstico fino de rolamento (ver limitação de resolução em `01_interpretacao_problema.md`, seção 1.4).
- **RD-04**: um sinal de rotação (RPM/encoder) para permitir *order tracking*, hoje impossível com os dados fornecidos.
- **RD-05**: acesso de leitura ao banco de dados já modelado pelo time de software do cliente (não a um novo *dump* de arquivos `.npy`), para que o pipeline de ingestão (`scripts/download_data.py` é um substituto temporário) leia direto da fonte.
- **RD-06**: um conjunto de dados com prior de classes realista (ou pelo menos rótulos de quando cada falha real ocorreu em produção), para validar a reamostragem sintética feita em `src/pdm/evaluation/imbalance.py` contra a realidade.
- **RD-07**: documentação existente de manutenção/RCA (*root cause analysis*) que a equipe de manutenção já usa (nenhuma parceria de IA industrial deveria ignorar décadas de conhecimento tácito do time de chão de fábrica sobre como cada falha se manifesta).

## 2.4 Requisito "plus": APIs e banco de dados

O enunciado cita como diferencial a implementação de APIs e bancos de dados, mas também deixa claro que o banco de dados de sensores **já existe**, modelado e implementado pelo time de software do cliente. Interpretamos isso como um convite a integrar, não duplicar:

- A **API** (`src/pdm/serving/api.py`, FastAPI) expõe o modelo como um serviço de inferência (`/predict`, `/predict_batch`), mais os serviços de confiança que o diferenciam de um simples wrapper de `model.predict()` (`/audit`, `/explain`, `/drift`, `/health`).
- O **banco de dados** deste projeto (`src/pdm/serving/db.py`) armazena apenas o que só a camada de IA produz (predições, relatórios de auditoria, relatórios de *drift*), nunca uma cópia do histórico bruto de sensores, que continua sendo responsabilidade e propriedade do time de software/automação. Em produção, a etapa de ingestão troca o `scripts/download_data.py` (que hoje lê do Google Drive do case) por um adaptador que lê diretamente do banco do cliente (o ponto de entrada do pipeline (`src/pdm/data/loader.py`) já é isolado o suficiente para essa troca ser localizada).
- Isso também é um requisito em aberto a validar com o cliente (RD-05 acima): a arquitetura assume acesso de leitura a esse banco; o formato exato (SQL relacional, *data lake*, *streaming*) muda o adaptador de ingestão, mas não o resto do pipeline.
- **Alto volume, preparado desde já**: o mesmo adaptador de ingestão que substitui `scripts/download_data.py` deve suportar leitura incremental (baseada em *timestamp* ou *change data capture*, não uma varredura completa da tabela a cada lote) e, quando o volume de sensores justificar, publicação em uma fila de mensagens (Kafka, RabbitMQ ou equivalente) entre a captação em campo e o pipeline de features, para que picos de throughput não bloqueiem a API de inferência. A própria API já expõe `/predict_batch` para consumo em lote; um *worker* de ingestão contínua reaproveitaria esse mesmo caminho de código, só trocando a origem dos dados (requisição HTTP síncrona hoje, mensagens de uma fila amanhã), sem duplicar a lógica de predição.
