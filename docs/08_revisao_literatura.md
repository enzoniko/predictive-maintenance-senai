# 8. Revisão de literatura (breve)

Revisão curta, com propósito de fundamentar as escolhas de engenharia desta prévia e as hipóteses do track de pesquisa (`06_track_pesquisa.md`) — não uma revisão sistemática. Organizada por linha de trabalho, com ênfase em manutenção preditiva de máquinas rotativas.

## 8.1 Features clássicas de diagnóstico de máquinas rotativas

O conjunto de indicadores estatísticos no domínio do tempo usado nesta prévia (RMS, fator de crista, curtose, fator de forma, fator de impulso — ver `src/pdm/features/time_domain.py`) e as bandas de energia no domínio da frequência (`frequency_domain.py`) são o vocabulário padrão de monitoramento de condição desde os anos 1990: curtose e fator de crista são sensíveis a impulsos característicos de defeitos localizados de rolamento; a análise espectral separa componentes harmônicos ligados a desbalanceamento, desalinhamento e frequências de passagem de elemento rolante. A limitação central desses métodos — a necessidade de resolução espectral fina e, idealmente, de um sinal de rotação para *order tracking* — é exatamente a limitação estrutural identificada nos dados deste case (janela de 20 ms, sem RPM; ver `01_interpretacao_problema.md`, seção 1.4, e G1 em `06_track_pesquisa.md`).

## 8.2 Análise tempo-frequência e wavelets

Transformadas wavelet contínuas (CWT) e suas variantes *synchrosqueezed* (SSQ-CWT) melhoram a localização tempo-frequência em relação à FFT de janela fixa, especialmente útil para sinais não estacionários — o regime típico de uma máquina sob carga variável. Esta prévia usa CWT via PyWavelets (`src/pdm/features/wavelet.py`) como energia por escala; a extensão *synchrosqueezed* (`ssqueezepy`) foi implementada e testada, mas só validada na plataforma Linux (ver `03_arquitetura.md`, seção 3.6) — fica registrada como próximo passo natural de refinamento (G5, `06_track_pesquisa.md`).

## 8.3 Aprendizado supervisionado clássico para diagnóstico de falhas

Métodos baseados em árvore (Random Forest, Gradient Boosting) sobre features de engenharia continuam competitivos com redes profundas em diagnóstico de falha quando o volume de dados rotulados por classe é moderado e as features já capturam a física relevante — exatamente o regime desta prévia (~50 mil janelas, ~150 features por sensor retido). Um ponto de comparação direto do próprio autor: em um benchmark de classificação de células cancerígenas em histopatologia (colaboração USP, 2025), XGBoost atingiu F1 = 0,88 no conjunto completo, enquanto redes Siamesas só superaram esse baseline em regime de dados escassos — reforçando a escolha desta prévia de partir do método mais simples que resolve o problema, e reservar arquiteturas mais pesadas para quando o regime de dados realmente exigir (ver H6/H7 em `06_track_pesquisa.md`).

## 8.4 Modelos informados por física (Physics-Informed Machine Learning)

Incorporar conhecimento físico como viés indutivo — via termos de perda ou restrições arquiteturais — reduz o espaço de hipóteses do modelo e melhora generalização fora da distribuição de treino, especialmente relevante quando dados de falha são raros. Duas linhas de trabalho do próprio autor sustentam essa aposta:

- **Bieger et al., "Physics-Informed Residual-Based Anomaly Detection and Open-Set Recognition System"**, IEEE IECON 2025 (prêmio IEEE IES SYPA): resíduos de uma PINN quantificam o desvio do comportamento esperado, um limiar é definido via Teoria de Valores Extremos, e uma rede Siamesa trata tanto falhas conhecidas quanto nunca vistas. Em sinais de vibração de máquina rotativa: 99,27% F1 de detecção, 82% de reconhecimento. Essa é a base direta das hipóteses H2 e H4 do track de pesquisa.
- **Bieger et al., "Linking Physical Fidelity to Downstream Performance in Physics-Informed Fault Diagnosis"**, IEEE Access (Q1), 2026: um protocolo de avaliação em três fases (fidelidade física, separabilidade, robustez *downstream*) mostra que baselines sem restrição física aprendem atalhos e colapsam em falhas nunca vistas, enquanto a perda física como viés indutivo atinge F1 = 84,0% de estado da arte.
- **Hard-Constrained Recurrent PINNs (HRPINN)** (tese de qualificação, UFSC, defendida perante banca UFSC/Coimbra/Luxemburgo, 2026): restrição física arquitetural em vez de penalizada, com melhor eficiência de dados e estabilidade de otimização — base da hipótese H8.

## 8.5 Aprendizado não, semi e auto-supervisionado

Em regime realista de planta industrial, dados saudáveis são abundantes e dados de falha são raros e frequentemente não cobrem todos os modos de falha possíveis — o oposto do conjunto balanceado deste case. A literatura de detecção de anomalia (*one-class classification*, autoencoders de reconstrução) e de reconhecimento *open-set* (redes Siamesas/métricas com limiar, como em Bieger et al. IECON 2025 acima) endereça exatamente esse descompasso, assim como pré-treino auto-supervisionado (contrastivo ou por reconstrução mascarada) sobre o grande volume de dados saudáveis não rotulados, com *fine-tuning* supervisionado leve nos poucos exemplos de falha disponíveis. Essas linhas compõem as hipóteses H3, H6 e H7 do track de pesquisa e não estão implementadas nesta prévia — o volume de dados de falha do case (10 mil janelas por classe) não representa esse regime de escassez, então o comportamento adequado nesta etapa é demonstrar o método supervisionado bem executado, não forçar um método pensado para um problema diferente.

## 8.6 Predição conformal para confiabilidade operacional

Predição conformal (Vovk et al.; Romano et al., 2020, *adaptive prediction sets*) fornece conjuntos de predição com garantia de cobertura livre de distribuição, sem exigir que o modelo subjacente seja bayesiano ou especialmente calibrado. É a base do requisito de "saber quando não sabe" desta prévia (`src/pdm/evaluation/conformal.py`) e generaliza diretamente para monitoramento contínuo em produção (H5, `06_track_pesquisa.md`), onde a cobertura empírica pode ser recalculada em janelas móveis como um segundo sinal de saúde do próprio modelo, complementar ao monitor de *drift* de features.

## 8.7 Onde este projeto se posiciona

A escolha de método desta prévia — features de engenharia interpretáveis + ensemble de árvores + predição conformal — não é a escolha "mais simples possível" por falta de conhecimento das alternativas mais recentes; é a escolha correta para o regime de dados e o requisito de confiança deste problema específico, com um roteiro de pesquisa explícito (physics-informed, open-set, auto-supervisionado) para quando o projeto contratado tiver os dados e o tempo para justificar a complexidade adicional.
