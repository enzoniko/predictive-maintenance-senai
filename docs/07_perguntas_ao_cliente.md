# 7. Perguntas em aberto para o cliente

Lista consolidada das lacunas que a auditoria de dados e a interpretação do problema não conseguiram resolver sozinhas (o tipo de pergunta que só o time de automação, instrumentação e manutenção do cliente pode responder). Cada uma está referenciada de onde surgiu no restante da documentação, para que a origem da dúvida fique rastreável.

## Sobre os dados fornecidos

1. **O enunciado descreve "seis arquivos .npy", mas o pacote contém `Classes.npy` + 5 arquivos `Dados_N.npy`.** Falta um sensor, ou a contagem de seis já incluía o arquivo de rótulos? (`01_interpretacao_problema.md`, seção 1.3)
2. **O que é a 201ª coluna presente em `Dados_1.npy`, `Dados_2.npy` e `Dados_3.npy`?** É quase inteiramente vazia (um único valor não-NaN por arquivo, sem relação consistente com a classe): descartamos como artefato de exportação, mas gostaríamos de confirmar a causa (índice perdido? marcador de fim de lote? erro de concatenação?). (`data/loader.py`, `data/audit.py`)
3. **Qual é a grandeza física de cada um dos 5 sensores** (vibração/aceleração, corrente, tensão, temperatura...), a sensibilidade do transdutor e o ganho do condicionador de sinal? Sem isso, todas as features permanecem em volts brutos, não em unidades de engenharia. (`configs/sensors.yaml`, `preprocessing/dsp.py::UnitConverter`)
4. **`Dados_4.npy` está travado num valor quase constante (~50) e `Dados_5.npy` é estatisticamente indistinguível de ruído branco em relação às classes.** São sensores com defeito real na aquisição, ou canais que simplesmente não se aplicam a este experimento (ex.: setpoint fixo, canal reservado)? (`data/audit.py`, resultado dos testes estatísticos)
5. **Os dados de todos os sensores foram realmente adquiridos de forma simultânea?** Não há timestamp nos arquivos para verificar diretamente; usamos um proxy (correlação de envoltória RMS entre sensores) que é compatível com essa afirmação, mas não a confirma. Existe um log de aquisição que permitiria validar isso com precisão? (`data/audit.py`, seção de simultaneidade)
6. **As 5 classes representam, de fato, diferentes tipos de falha, diferentes níveis de severidade da mesma falha, ou diferentes regimes de operação saudável?** O enunciado deixa essa semântica deliberadamente aberta ("podendo ou não possuir diferentes falhas e/ou funcionamento normal"), mas o projeto contratado precisa dela para transformar "classificar padrões" em "diagnosticar falhas". (`02_engenharia_requisitos.md`, RD-02)
7. **Por que as 5 classes estão perfeitamente balanceadas (10.000 janelas cada)?** Isso não costuma ocorrer naturalmente numa planta em operação. Foi uma amostragem deliberada para fins do processo seletivo? Qual é a proporção real observada em produção? (`evaluation/imbalance.py`, RD-06)
8. **Existe um sinal de rotação (RPM/encoder) disponível, mesmo que não incluído neste pacote?** Sem ele, técnicas de *order tracking* e análise fina de frequência de falha de rolamento não são viáveis. (`01_interpretacao_problema.md`, seção 1.4; RD-04)
9. **A taxa de amostragem de 10 kHz e a janela de 200 amostras (20 ms) são as mesmas usadas na aquisição de produção, ou foram reduzidas/recortadas especificamente para este case?** Um *stream* contínuo com janelas maiores mudaria significativamente o que é tecnicamente viável na Fase 3 (pesquisa avançada). (RD-03)

## Sobre o motor/máquina em si

10. **Qual é o tipo específico de motor elétrico** (indução trifásica, síncrono, corrente contínua...), potência, velocidade nominal e tipo de rolamento/acoplamento? Necessário para qualquer abordagem *physics-informed* (H1, H2, H8 em `06_track_pesquisa.md`).
11. **Que tipos de falha esse motor já apresentou historicamente**, e o time de manutenção já realiza análise de causa raiz (RCA) documentada para elas? Se sim, essa documentação é um dos ativos mais valiosos que o projeto poderia herdar (décadas de conhecimento tácito não deveriam ser reaprendidas do zero por um modelo). (RD-07)
12. **Onde fisicamente os sensores estão instalados na máquina** (carcaça, mancal, próximo a qual componente)? Afeta diretamente quais frequências características são fisicamente esperadas em cada canal.

## Sobre integração e requisito "plus" de APIs/bancos de dados

13. **Qual é a natureza exata do banco de dados já modelado pelo time de software** (SQL relacional, série temporal dedicada, *data lake*)? Determina o adaptador de ingestão que substituirá `scripts/download_data.py` na fase contratada. (`02_engenharia_requisitos.md`, seção 2.4, RD-05)
14. **Existe uma expectativa de latência de resposta** para a API de inferência em produção (tempo real por máquina, ou processamento em lote periódico)? Não medido formalmente nesta prévia (RNF-04 assume folga grande dado o modelo raso escolhido).
15. **Existe um sistema de alarme/dashboard de chão de fábrica já em uso** ao qual as predições devem se integrar, ou o dashboard construído nesta prévia (Streamlit) é o ponto de partida esperado?

## Sobre avaliação e critérios de sucesso do projeto contratado

16. **Qual é a definição de sucesso do cliente** para o piloto (Fase 4 em `04_cronograma_9_meses.md`): número de falsos alarmes toleráveis, tempo de antecedência mínimo de um alarme antes da falha real, ou outro critério de negócio específico?
