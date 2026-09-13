# 4. Arquitetura de projeto e cronograma de execução (caso contratado)

## 4.1 Premissas

- Escopo: **uma máquina/motor específico**, com os requisitos de dados de [`02_engenharia_requisitos.md`](02_engenharia_requisitos.md) (RD-01 a RD-07) atendidos até o fim da Fase 0.
- Equipe: 1 mentor/líder técnico de IA (tempo parcial, decisões de arquitetura e revisão), 2 engenheiros de ML/dados (tempo integral), 1 engenheiro de integração/backend (tempo parcial a partir da Fase 3), apoio pontual do time de automação/manutenção do cliente.
- Duração total: **9 meses**. Esse prazo é deliberadamente contido: para uma solução especializada numa única máquina, com um time bem mentorado e os requisitos de dados atendidos, um cronograma mais longo indicaria escopo mal definido, não rigor.
- O que muda esse prazo: se os requisitos de dados (RD-01–RD-07) não forem atendidos na Fase 0, ou se o cliente quiser escalar para múltiplas máquinas/tipos de ativo dentro do mesmo contrato, o cronograma deve ser renegociado (não espremido).

## 4.2 Fases

```mermaid
gantt
    title Cronograma do projeto de manutencao preditiva (9 meses)
    dateFormat  YYYY-MM-DD
    axisFormat  %b
    section Fase 0 (Fundacao)
    Kickoff e coleta de requisitos      :f0a, 2026-10-01, 20d
    Acesso ao banco de dados do cliente  :f0b, after f0a, 15d
    section Fase 1 (Entendimento)
    Revisao de literatura e dominio      :f1a, after f0b, 15d
    Auditoria de dados no banco real     :f1b, after f0b, 20d
    section Fase 2 (Baseline)
    Engenharia de features fisicas       :f2a, after f1b, 20d
    Modelo interpretavel baseline        :f2b, after f2a, 20d
    Avaliacao e validacao com o cliente  :f2c, after f2b, 10d
    section Fase 3 (Integracao)
    API + banco de producao              :f3a, after f2c, 20d
    Dashboard e alertas                  :f3b, after f3a, 15d
    Piloto em 1 maquina                  :f3c, after f3b, 20d
    section Fase 4 (Validacao de campo)
    Operacao assistida e ajustes         :f4a, after f3c, 30d
    Transferencia de conhecimento        :f4b, after f4a, 15d
    section Fase 5 (Acompanhamento)
    Monitoramento de drift e retrain     :f5a, after f4b, 30d
```

## 4.3 Estrutura analítica do projeto (WBS resumida)

1. **Fase 0** (Kick-off e requisitos, mês 1)
   - Reunião de descoberta com automação, manutenção e TI do cliente.
   - Fechamento formal de RD-01 a RD-07 (`02_engenharia_requisitos.md`).
   - Acesso de leitura ao banco de dados existente (não a novos arquivos `.npy`).
   - **Marco M0**: requisitos de dados assinados pelas partes; acesso ao banco liberado.
2. **Fase 1** (Entendimento do domínio, mês 2)
   - Revisão de literatura de diagnóstico da máquina/família de falha específica (ver `08_revisao_literatura.md`).
   - Reexecução da auditoria de dados (`src/pdm/data/audit.py`) contra o volume real do banco do cliente, não a amostra curada do case.
   - **Marco M1**: relatório de auditoria + lista de sensores validados/descartados para essa máquina específica.
3. **Fase 2** (Baseline interpretável, meses 3-4)
   - Engenharia de features com significado físico (unidades reais, via `UnitConverter` já preparado em `preprocessing/dsp.py`, agora com sensibilidades reais).
   - Modelo baseline (árvore/ensemble), replicando a metodologia desta prévia com os dados reais e prior de classes real.
   - Validação cruzada com o time de manutenção: as classes/estados batem com o que eles observam em campo?
   - **Marco M2**: baseline aprovado pelo cliente como "digno de confiança para operar em modo sombra".
4. **Fase 3** (Integração e piloto, meses 5-6)
   - API de inferência e banco de dados de predição/auditoria/drift em produção (arquitetura já prototipada nesta prévia, `src/pdm/serving/`).
   - Dashboard para o time de manutenção (extensão do protótipo Streamlit ou integração com o BI já usado pelo cliente).
   - Piloto controlado em uma máquina real, em paralelo ao processo de manutenção atual (não substituindo-o ainda).
   - **Marco M3**: piloto rodando em produção-sombra por ≥ 30 dias sem incidentes de infraestrutura.
5. **Fase 4** (Validação de campo e transferência, mês 7)
   - Comparação sistemática entre alarmes do modelo e eventos reais de manutenção/falha.
   - Treinamento do time interno do cliente (operação, leitura de alarmes, quando escalar para o time de IA).
   - **Marco M4**: aceite formal do cliente para operação assistida (não mais sombra).
6. **Fase 5** (Acompanhamento e evolução, meses 8-9 e contrato de sustentação a partir daí)
   - Monitoramento contínuo de *drift* (`evaluation/drift.py` já implementado), gatilhos de retrain (ver `09_mlops.md`).
   - Ciclo de melhoria contínua incorporando o que a trilha de pesquisa interna (seção 4.5) validar como pronto para produção.

## 4.4 Esforço estimado (pessoa-mês, ordem de grandeza)

| Fase | Duração | Líder técnico | Eng. ML/dados (x2) | Eng. integração |
|---|---|---|---|---|
| 0 (Requisitos) | 1 mês | 0,3 | 0,5 | N/A |
| 1 (Entendimento) | 1 mês | 0,2 | 1,0 | N/A |
| 2 (Baseline) | 2 meses | 0,4 | 2,0 | N/A |
| 3 (Integração) | 2 meses | 0,3 | 1,0 | 1,0 |
| 4 (Validação) | 1 mês | 0,3 | 0,5 | 0,5 |
| 5 (Acompanhamento, mês 8+) | contínuo | 0,1/mês | 0,3/mês | 0,2/mês |

Esse esforço é a base para a proposta comercial que acompanha este documento nos slides executivos (não é repetido aqui em valores monetários porque esses dependem de taxas contratuais do SENAI SC que não fazem parte do escopo técnico desta prévia).

## 4.5 Trilha de pesquisa interna (equipe de IA do SENAI, não contratada pelo cliente)

O cronograma acima entrega o que uma indústria contratando manutenção preditiva efetivamente precisa: um baseline interpretável, validado com o time de manutenção, integrado a API e banco de dados, rodando em piloto e depois em operação assistida. Nenhuma dessas fases depende de pesquisa avançada para acontecer, e é assim de propósito: o enunciado deste processo seletivo nunca pede estado da arte ou publicação, pede que o diretor da empresa seja convencido a contratar o projeto. Colocar uma fase de pesquisa no meio do cronograma do cliente, bloqueando ou disputando recursos com a entrega, resolveria um problema que o cliente não tem.

Isso não significa abrir mão da pesquisa. Como Tech Lead, a decisão de manter uma trilha de pesquisa (physics-informed models, aprendizado não/semi/auto-supervisionado, reconhecimento *open-set*, predição conformal recalibrada em produção, ver `06_track_pesquisa.md`) é deliberada e paralela, não cortada: ela custeia 1 ML engineer adicional por cerca de 3 meses (meses 3 a 6, correndo junto com as Fases 2 e 3 acima), como uma iniciativa interna da equipe de IA do SENAI. O critério de promoção continua o mesmo: uma técnica de pesquisa só substitui um componente do pipeline de produção se ganhar em robustez ou generalização documentada **e** não piorar a interpretabilidade ou a latência abaixo dos requisitos não-funcionais (`02_engenharia_requisitos.md`). Se e quando uma hipótese passar nesse critério, ela entra no ciclo de melhoria contínua da Fase 5, sem nunca ter sido uma dependência do que foi vendido ao cliente.

Essa separação entre cronograma do cliente e trilha de pesquisa interna é, em si, parte do que se espera de um mentor técnico sênior: entender o que a empresa precisa (o problema resolvido, com confiança e prazo) e, ao mesmo tempo, manter viva a capacidade de pesquisa do time, sem confundir as duas coisas nem impor o custo de uma à outra.
