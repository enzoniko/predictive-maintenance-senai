# Roteiro de vídeo 1 — Diretor não técnico (executivo)

**Duração alvo**: 3–5 min. **Formato**: conversa simulada (Enzo interpreta os dois papéis — Mentor de IA e Diretor — ou grava só o papel de Mentor e legenda as falas do Diretor). **Apoio visual**: `Apresentacao_Executiva.pdf`, slides 1–6.

**Objetivo da cena**: o diretor não quer detalhes técnicos — quer saber se vale a pena contratar, com que risco, e em quanto tempo. A cena termina com o diretor pedindo a proposta formal.

---

**[Diretor]**: Antes de eu autorizar esse projeto, me explica em uma frase: isso funciona nos meus dados?

**[Mentor]**: Funciona. Com os sensores que vocês já têm instalados nessa máquina, o modelo identifica corretamente o estado de operação em mais de 96% dos casos de teste — e o mais importante: quando ele não tem certeza, ele avisa, em vez de arriscar um palpite.

*(Mostrar slide com a matriz de confusão e o número 96%.)*

**[Diretor]**: "Avisa quando não tem certeza" — o que isso quer dizer na prática?

**[Mentor]**: Em vez de só dizer "é falha tipo B", o sistema pode dizer "é falha tipo B ou C, não consigo distinguir com segurança — chama um técnico para confirmar". Isso é o que chamamos de predição com incerteza calibrada. É a diferença entre um sistema que finge saber tudo e um que sabe quando não sabe. Para uma decisão que pode envolver parar uma linha de produção, essa diferença vale mais do que um número de acurácia bonito.

*(Mostrar slide com exemplo de conjunto de predição: "{Falha B, Falha C} — confiança insuficiente para decidir sozinho".)*

**[Diretor]**: E se os dados que vocês usaram forem bons demais, artificiais? Isso vai funcionar de verdade na minha fábrica?

**[Mentor]**: Essa é a pergunta certa, e é literalmente a primeira coisa que fizemos antes de treinar qualquer modelo. Auditamos os dados que vocês nos enviaram e encontramos, por exemplo, um dos cinco sensores completamente travado e outro que era só ruído — nós mesmos excluímos os dois, com teste estatístico, antes de treinar. E as cinco classes que vocês nos mandaram estão artificialmente balanceadas, o que não acontece numa fábrica de verdade — então já reavaliamos o modelo simulando a proporção real que vocês devem ter em produção, não só o cenário perfeito.

*(Mostrar slide "o que a auditoria encontrou" com os 2 sensores excluídos.)*

**[Diretor]**: Quanto tempo até isso rodar de verdade, numa máquina real?

**[Mentor]**: Nove meses até o piloto validado em produção-sombra, com uma máquina específica bem definida. Isso inclui um mês inicial só para alinhar com o seu time de automação o que cada estado da máquina realmente significa — sem isso, o sistema classifica padrões, mas não te diz "isso é uma falha de rolamento", só "isso é diferente do normal". Essa conversa inicial é o item que mais influencia se o prazo de nove meses se sustenta.

*(Mostrar o cronograma resumido, 4 fases principais.)*

**[Diretor]**: E o risco de isso não dar certo?

**[Mentor]**: Cada risco que identificamos já tem um mecanismo de mitigação funcionando nesta prévia, não uma promessa — a auditoria de dados, a detecção de quando o modelo está "desatualizado" em relação à realidade da máquina, a explicação de cada decisão. Não estamos pedindo para você confiar numa caixa preta.

**[Diretor]**: Combinado. Manda a proposta formal com o cronograma e a equipe.

**[Mentor]**: Já está nos anexos.

---

**Notas de produção**: manter tom direto, sem jargão técnico não explicado (evitar "F1-macro", "conformal", "HistGradientBoosting" faladas sem tradução — usar "acerto", "avisa quando não sabe", "modelo escolhido"). Cortar qualquer resposta acima de 30 segundos.
