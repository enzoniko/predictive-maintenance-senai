# 9. MLOps

Como o modelo, os dados e a infraestrutura deste projeto são versionados, monitorados e evoluídos ao longo do tempo — não um artefato entregue uma vez e esquecido.

## 9.1 Versionamento

| O que | Como |
|---|---|
| Código | Git (este repositório), commits progressivos, CI (`.github/workflows/ci.yml`) rodando lint + testes em Python 3.10/3.11/3.12 a cada push |
| Configuração | `configs/default.yaml` e `configs/sensors.yaml`, versionados junto com o código — nenhum limiar mágico vive só na cabeça de quem treinou o modelo |
| Dados | Fora do repositório (`.gitignore`), buscados via `scripts/download_data.py`; em produção, isso vira um adaptador de leitura do banco do cliente (RD-05, `02_engenharia_requisitos.md`) com um identificador de versão/*snapshot* do lote usado em cada treino |
| Experimentos | `models/registry.py`: MLflow quando disponível na plataforma, ou um *tracker* local (JSON + joblib) como fallback — ver `03_arquitetura.md`, seção 3.6, para por que essa dualidade existe. Cada execução registra parâmetros, métricas e o artefato do modelo |
| Modelo em produção | `ModelBundle` (`models/bundle.py`): um único artefato versionável contendo o modelo, os limpadores de sensor ajustados, o calibrador conformal e metadados (nome do modelo, colunas de feature, classes, timestamp de criação) — nunca partes soltas que podem dessincronizar entre si |

## 9.2 Pipeline de CI/CD (o que já existe vs. o que falta para produção)

- **Já existe** (`​.github/workflows/ci.yml`): lint (`ruff`) + suíte de testes (`pytest`) contra um fixture sintético, a cada push, em três versões de Python.
- **Falta para produção** (Fase 4, `04_cronograma_9_meses.md`): passo de *build* + push de imagem Docker (`docker/Dockerfile.api`, `Dockerfile.dashboard`) para um registro de contêineres, e um passo de *deploy* (rolling update na API, nunca substituição abrupta — ver 9.4).
- **Gate de qualidade para promoção de modelo**: nenhum modelo novo substitui o `ModelBundle` em produção sem passar pelos mesmos controles de sanidade desta prévia (embaralhamento de rótulo próximo do acaso, F1 de teste acima de um piso mínimo acordado com o cliente, cobertura conformal dentro de uma banda aceitável do alvo configurado).

## 9.3 Monitoramento em produção

Dois sinais de saúde independentes, ambos já implementados nesta prévia:

1. **Drift de dados** (`evaluation/drift.py`, `GET /drift`): PSI + Kolmogorov-Smirnov por feature, comparando o lote de produção recente contra a distribuição de treino. Um PSI acima do limiar configurado (`configs/default.yaml: drift.psi_alarm_threshold`) para qualquer feature dispara alarme. **Ressalva observada na prática** (`notebooks/03_api_demo.ipynb`): com poucas amostras no lote atual, o PSI fica instável e dispara alarmes só por ruído amostral -- comparar ~40 janelas (mesmo vindas exatamente da mesma distribuição da referência) contra uma referência de 50 mil já produziu 143 de 176 features "em alarme", um falso positivo generalizado, não uma medição. Por isso `GET /drift` (`serving/api.py`) recusa (503) calcular um score com menos de `MIN_PRODUCTION_SAMPLES_FOR_DRIFT` (200) predições registradas -- um piso mais defensável, ainda assim uma regra prática, não um substituto para revisar o relatório de drift real em produção.
2. **Cobertura conformal empírica** (H5, `06_track_pesquisa.md`): a cobertura real observada numa janela móvel de predições com rótulo verdadeiro disponível (via feedback do time de manutenção) deve se manter próxima do alvo configurado; um desvio sistemático é, em si, um sinal de que o modelo ou a calibração precisam de atenção, antes mesmo de a acurácia cair visivelmente.

Toda predição, auditoria e relatório de *drift* fica persistido (`serving/db.py`), o que torna esses dois sinais consultáveis retroativamente, não só como alarme em tempo real.

## 9.4 Estratégia de retrain e rollback

- **Gatilhos de retrain**: alarme de *drift* sustentado por mais de N dias (configurável), cobertura conformal fora da banda aceitável, ou um ciclo calendário mínimo (ex.: trimestral) independente de alarme, para acompanhar desgaste natural do equipamento.
- **Sem downtime**: o novo `ModelBundle` é validado offline contra o gate de qualidade (9.2) antes de qualquer substituição; a API recarrega o bundle sem precisar ficar fora do ar (troca do arquivo + *reload* controlado, ou *blue/green* com duas instâncias da API apontando para bundles diferentes até a validação em produção-sombra ser aprovada).
- **Rollback**: como cada `ModelBundle` é um artefato único e versionado (9.1), reverter para a versão anterior é uma troca de arquivo, não um retrain de emergência.

## 9.5 Segurança e integridade dos dados

- `Classes.npy` é um array `dtype=object` que exige `allow_pickle=True` para ser carregado — um risco de cadeia de suprimentos de dados (um arquivo trocado poderia executar código arbitrário na desserialização). O `loader` (`src/pdm/data/loader.py`) converte para um array de strings seguro imediatamente após o carregamento e nenhum outro módulo toca a forma *pickled* novamente. Em produção, a fonte de rótulos deveria vir do banco relacional do cliente, eliminando esse vetor por completo.
- O banco de dados desta camada de IA (`serving/db.py`) nunca duplica dados brutos de sensores do cliente — só armazena o que a própria IA produz (predições, auditorias, relatórios de *drift*), reduzindo a superfície de exposição de dados sensíveis da planta.

## 9.6 Ambiente e reprodutibilidade entre plataformas

Este projeto documenta e testa explicitamente a diferença entre o ambiente de desenvolvimento (Windows ARM64) e o ambiente de implantação alvo (Linux x86-64) — ver `03_arquitetura.md`, seção 3.6. Isso não é incidental ao MLOps: um pipeline que "funciona na minha máquina" e falha silenciosamente em produção é um dos modos de falha mais comuns e mais evitáveis em projetos de ML aplicado. A prática adotada aqui — nunca deixar uma dependência ausente quebrar o pipeline, sempre com *fallback* documentado e testado nas duas plataformas — é o que este projeto propõe como padrão para todo o portfólio de manutenção preditiva do cliente, não só para esta máquina específica.
