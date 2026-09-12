# Roteiro de vídeo 3 — Engenheiro de automação/instrumentação (técnico)

**Duração alvo**: 5 min. **Apoio visual**: `Apresentacao_Tecnica.pdf`, terminal com `python -m pdm.cli audit`, notebook `01_data_audit.ipynb`.

**Objetivo da cena**: o engenheiro de automação instalou os sensores e sabe exatamente o que eles deveriam medir — é o público que vai validar (ou derrubar) nossas hipóteses sobre os dados e definir a integração real.

---

**[Engenheiro]**: Vocês receberam só os arquivos `.npy`, sem nenhuma documentação nossa sobre os sensores. Como validaram alguma coisa?

**[Mentor]**: Não validamos tudo — documentamos exatamente o que não dava para validar sem vocês, e tratamos como pergunta em aberto, não como suposição silenciosa. *(Abre `docs/07_perguntas_ao_cliente.md`.)* Por exemplo: os arquivos `Dados_1`, `2` e `3` têm uma coluna a mais do que o esperado, quase inteiramente vazia — a gente descartou, mas registrou três hipóteses do que pode ter causado isso, porque vocês provavelmente sabem a resposta de cara.

**[Engenheiro]**: E os outros dois arquivos, `Dados_4` e `Dados_5`?

**[Mentor]**: `Dados_4` está travado num valor quase constante — ou é um sensor com defeito, ou é um canal que não se aplica a esse experimento, tipo um setpoint fixo. `Dados_5` é estatisticamente indistinguível de ruído branco em relação aos estados da máquina — testamos com Kruskal-Wallis e um teste de permutação de informação mútua, não foi "parece ruído no olho". Excluímos os dois do modelo, com justificativa estatística, mas a causa raiz só vocês confirmam.

*(Mostra o terminal rodando `python -m pdm.cli audit` e o resultado.)*

**[Engenheiro]**: Vocês assumiram que os sinais são vibração?

**[Mentor]**: Assumimos como hipótese de trabalho, pelo conteúdo espectral e pela saturação por volta de ±5 V, típica de um condicionador — mas o pipeline inteiro roda em volts brutos, sem converter para g ou mm/s, porque não temos a sensibilidade do transdutor. *(Mostra `configs/sensors.yaml`.)* Assim que vocês confirmarem o tipo de sensor e a sensibilidade, é uma linha de configuração, não uma reescrita de código.

**[Engenheiro]**: Como isso vai se conectar no nosso banco de dados, sem vocês ficarem pedindo arquivo `.npy` toda semana?

**[Mentor]**: A arquitetura já separa isso — hoje, a etapa de entrada lê do Google Drive do processo seletivo; em produção, ela vira um adaptador que lê direto do banco de vocês. O resto do pipeline não muda. E o banco que a gente constrói não duplica o histórico bruto de sensores, que é de vocês — só guarda o que a camada de IA produz: predições, auditorias, alertas de mudança de comportamento dos dados.

*(Mostra o diagrama de arquitetura, `docs/03_arquitetura.md`.)*

**[Engenheiro]**: Falta um sinal de rotação, RPM. Isso importa?

**[Mentor]**: Importa bastante, principalmente pra fase de pesquisa mais avançada — sem RPM, a gente não consegue separar direito frequências de falha específica de rolamento, só classificar o padrão geral. Está no requisito de dados da Fase 0, junto com a semântica de cada classe.

**[Engenheiro]**: Ok, isso é uma conversa de uma tarde com o meu time. Bora marcar.

---

**Notas de produção**: este é o vídeo mais técnico dos três — pode e deve usar termos de instrumentação corretamente (sensibilidade, condicionador, ADC, fundo de escala). Priorizar tela do terminal/notebook sobre rosto.
