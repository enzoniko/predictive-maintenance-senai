# Roteiro de vídeo 2 — Gerente de manutenção (operacional)

**Duração alvo**: 4–5 min. **Apoio visual**: dashboard Streamlit (`src/pdm/serving/dashboard.py`) rodando ao vivo, ou capturas de tela.

**Objetivo da cena**: o gerente de manutenção vai operar o sistema no dia a dia — ele não quer saber de arquitetura, quer saber se pode confiar no alarme e o que fazer quando ele aparece.

---

**[Gerente]**: Se esse sistema me manda um alarme às 3h da manhã, eu mando alguém parar a máquina?

**[Mentor]**: Depende do que o alarme diz — e ele foi desenhado para não te dar só um "sim ou não". Deixa eu te mostrar na tela. *(Abre o dashboard, faz uma predição de exemplo.)* Aqui, o sistema classificou como "Classe B" com alta confiança, e o conjunto de possibilidades é só essa classe — isso é um alarme forte. Já aqui *(segundo exemplo, com conjunto de duas classes)*, o sistema não consegue distinguir entre B e C — isso é "chama o técnico para inspecionar", não "para a máquina agora".

**[Gerente]**: E se o sinal do sensor estiver ruim naquele momento, o sistema finge que não viu?

**[Mentor]**: Não — ele avisa isso explicitamente. *(Mostra o aviso "janela sinalizada como quase silenciosa por pelo menos um sensor".)* Nós já vimos isso acontecer nos dados que vocês nos passaram: por volta de 1 a 2% das janelas de cada sensor real vêm quase sem sinal, provavelmente um sensor com folga momentânea ou a máquina realmente parada. Em vez de o modelo inventar uma resposta com dado ruim, ele sinaliza a situação.

**[Gerente]**: Como eu sei que esse sistema não vai "esquecer" como a máquina se comporta com o tempo, depois que ela passar por uma manutenção?

**[Mentor]**: Isso é monitorado automaticamente — o sistema compara constantemente os dados novos com os dados que ele usou pra aprender, e se a diferença ficar grande demais, ele acende um alarme de "isso mudou, hora de retreinar", antes mesmo da taxa de erro subir visivelmente. Isso já está rodando no protótipo, na aba de auditoria do painel.

*(Mostra a seção de auditoria/drift do dashboard.)*

**[Gerente]**: E se eu quiser entender *por que* ele decidiu aquilo, não só o quê?

**[Mentor]**: Cada predição vem com as variáveis que mais pesaram na decisão — não é uma caixa preta que só cospe um rótulo. *(Mostra a explicação por predição.)* Com o tempo, à medida que vocês forem confirmando os alarmes reais, a gente ajusta os limiares junto com o time de vocês, não por fora dele.

**[Gerente]**: Combinado, quero ver isso rodando em paralelo com o que a gente já faz antes de eu confiar 100%.

**[Mentor]**: É exatamente assim que o piloto está desenhado — em sombra, ao lado do processo atual, antes de qualquer substituição.

---

**Notas de produção**: gravar com o dashboard realmente aberto (`streamlit run src/pdm/serving/dashboard.py`, com a API rodando); priorizar tela sobre rosto. Vocabulário: "alarme", "confiança", "sinal ruim", nunca "classe", "conformal" ou "F1" sem tradução.
