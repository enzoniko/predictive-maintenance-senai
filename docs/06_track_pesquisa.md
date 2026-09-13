# 6. Track de pesquisa (plano interno da equipe de IA do SENAI SC)

Este documento é interno à equipe de IA do SENAI SC, não parte do escopo contratado pelo cliente (ver `04_cronograma_9_meses.md`, seção 4.5, para onde essa trilha entra no cronograma e por quê). O cronograma e a proposta que vão para o cliente entregam um pipeline interpretável, validado e integrado, sem depender de nenhuma hipótese abaixo: quem contrata manutenção preditiva quer o problema resolvido, não necessariamente estado da arte publicável. Mantê-las aqui, separadas, é o que permite ao time de IA continuar avançando cientificamente sem transformar isso numa exigência ou num risco de prazo para o cliente.

Manutenção preditiva de máquinas rotativas é uma área madura, mas isso não significa que o problema deste cliente esteja resolvido. Significa que existe uma literatura sólida para não reinventar o básico, e uma fronteira ativa (aprendizado com poucos dados de falha, robustez a domínio, incerteza calibrada) onde ainda há ganho real de pesquisa. Este documento define os **gaps** identificados, as **hipóteses** de pesquisa associadas e o **protocolo experimental** para testá-las.

## 6.1 Gaps identificados

- **G1** (janela curta, sem RPM): 20 ms de janela e ausência de sinal de rotação impedem *order tracking* e separação fina de frequências de falha de rolamento (ver `01_interpretacao_problema.md`, seção 1.4). Qualquer técnica que dependa de resolução espectral fina para funcionar precisa ser revalidada quando dados de melhor resolução estiverem disponíveis.
- **G2** (regime de poucos dados de falha): o conjunto do case está artificialmente balanceado; uma planta real tem abundância de dados saudáveis e escassez de exemplos de cada tipo de falha. O pipeline desta prévia é supervisionado e assume rótulos abundantes para todas as classes, o que não se sustenta em produção sem adaptação.
- **G3** (falhas nunca vistas): um classificador fechado (softmax sobre classes conhecidas) não tem mecanismo nativo para dizer "isto não é nenhuma das classes que eu conheço".
- **G4** (ausência de modelo físico validado): sem confirmação do tipo exato de motor, rolamento ou acoplamento, não é possível hoje incorporar equações físicas específicas (frequências de falha teóricas, modelo de dinâmica rotativa) como viés indutivo.
- **G5** (wavelet/tempo-frequência subexplorado nesta prévia): a prévia usa CWT (PyWavelets) para energia por escala; a versão *synchrosqueezed* (`ssqueezepy`, ver `03_arquitetura.md` seção 3.6) foi implementada e validada, mas não integrada ao modelo final por restrição de tempo, ficando como extensão natural de curto prazo.

## 6.2 Hipóteses de pesquisa

| ID | Hipótese | Abordagem | Métrica de sucesso |
|---|---|---|---|
| H1 | Features físicas informadas (quando a grandeza/unidade for confirmada) melhoram generalização entre condições de operação em relação a features puramente estatísticas | *Physics-informed feature engineering*: integração aceleração→velocidade (ISO 20816), *order tracking* quando RPM disponível | Ganho de F1 em validação cruzada por condição operacional, não só por janela aleatória |
| H2 | Um termo de regularização física (PINN residual) aumenta a robustez a condições de operação não vistas em treino | Arquitetura *physics-informed residual*, na linha de Bieger et al., IECON 2025, ACCESS 2025 (ver `08_revisao_literatura.md`) | AUROC de detecção de anomalia em condições fora da distribuição de treino |
| H3 | Um detector *one-class* (treinado só em operação saudável) complementa o classificador supervisionado no regime de poucos dados de falha (G2) | IsolationForest / autoencoder de reconstrução / resíduo PINN + Extreme Value Theory para limiar | AUROC de detecção de "não-saudável" vs. saudável, sem usar rótulos de falha no treino |
| H4 | Uma rede Siamesa / métrica aprendida permite reconhecimento *open-set* de falhas nunca vistas em treino (G3) | Aprendizado de representação métrica + limiar por classe, na linha de Bieger et al., IECON 2025, ACCESS 2025 (99,27% F1 de detecção, 82% de reconhecimento em vibração de máquina rotativa) | Taxa de reconhecimento correto de uma classe de falha *holdout*, nunca vista em treino |
| H5 | Predição conformal calibrada em produção mantém cobertura nominal sob deslocamento de distribuição (drift) moderado | *Split conformal* recalibrado periodicamente com o monitor de drift (`evaluation/drift.py`) como gatilho | Cobertura empírica medida em janelas móveis de produção, comparada ao alvo configurado |
| H6 | Pré-treino auto-supervisionado (contrastivo ou de reconstrução mascarada) em dados saudáveis não rotulados melhora a eficiência de rótulo para as classes de falha raras | Pré-treino em `Dados_1-3` saudáveis sem rótulo, *fine-tuning* com poucos exemplos rotulados de falha | F1 em regime de poucos exemplos (*few-shot*), comparado ao baseline supervisionado direto |
| H7 | Propagação de rótulos semi-supervisionada reduz o custo de rotulagem manual de novas condições de falha | *Label propagation* / pseudo-rotulagem a partir de um pequeno núcleo rotulado | Redução de exemplos rotulados necessários para atingir o F1 do baseline totalmente supervisionado |
| H8 | Um modelo *Hard-Constrained Recurrent PINN* (HRPINN) generaliza melhor entre regimes operacionais do que um modelo sem restrição física, com melhor eficiência de dados | Adaptar a arquitetura de Bieger et al. (TCC graduação, UFSC, 2025) ao domínio de vibração/corrente deste projeto | Eficiência de dados (F1 em função do número de exemplos de treino) |

## 6.3 Protocolo experimental (comum a todas as hipóteses)

1. **Baseline fixo**: toda hipótese é comparada contra o modelo desta prévia (HistGradientBoosting sobre features de tempo/frequência/wavelet), não contra um baseline artificialmente fraco.
2. **Split por condição operacional**, não por janela aleatória, sempre que a informação de condição estiver disponível (evita que o modelo "decore" uma condição específica em vez de aprender o fenômeno).
3. **Controles obrigatórios**: toda hipótese reporta o mesmo par de controles já usado na prévia (embaralhamento de rótulo, canal isolado sem sinal) antes de qualquer alegação de ganho.
4. **Critério de promoção a produção**: uma técnica só substitui o componente correspondente do pipeline de produção se ganhar em robustez/generalização documentada **e** não piorar a interpretabilidade/latência abaixo dos requisitos não-funcionais (`02_engenharia_requisitos.md`).

## 6.4 Datasets públicos para viabilizar a pesquisa em paralelo (risco R1)

Enquanto os requisitos de dados do cliente (RD-01–RD-07) não são plenamente atendidos, a evolução das hipóteses H1–H8 não precisa parar: datasets públicos de máquinas rotativas com configuração similar (vibração/corrente, múltiplos estados de falha) permitem continuar desenvolvendo e validando o pipeline.

- **CWRU Bearing Data Center** (Case Western Reserve University): referência clássica de falha de rolamento.
- **MFPT** (Machinery Failure Prevention Technology): vibração de rolamento sob múltiplas cargas.
- **Paderborn University Bearing Dataset** (inclui falhas reais (não só induzidas), corrente e vibração).
- **MaFaulDa** (Machinery Fault Database, UFRJ): vibração e áudio de máquina rotativa com desbalanceamento, desalinhamento e falha de rolamento, útil especialmente por já estar em português/contexto brasileiro.
- **Multi-Speed Raw Triaxial Vibration Dataset for Rotating Machinery** (Regis Cardoso, SENAI, ISI-SE): Este conjunto de dados fornece sinais brutos de vibração triaxial adquiridos em um banco de testes de máquinas rotativas, sob operação normal e sete condições de falha induzidas fisicamente: desbalanceamento, desalinhamento, perda de fase e defeitos nos rolamentos — pista interna, elemento rolante (esfera), combinados e pista externa. :)

Essa é, em si, a mitigação prototipada do risco R1 em `05_riscos_e_mitigacoes.md`: o time de pesquisa não fica bloqueado esperando o cliente.
