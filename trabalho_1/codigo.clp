; DISCIPLINA:
;  - Inteligência Artificial (INE5430)
;
; ALUNOS:
;  - Augusto de Hollanda Vieira Guerner (22102192)
;  - Eduardo Dani Perottoni (21204003)
;
; PROFESSORA:
;  - Jerusa Marchi
;
; DOMÍNIOS MODELADOS:
;  - Doenças cardíacas comuns
;  - Sintomas das doenças apresentadas
;  - Tratamentos para as doenças apresentadas


; Doenças
(deftemplate Doenca
	(slot doenca))

; Tratamentos
(deftemplate Tratamento
	(slot tratamento))

; Sintomas
(deftemplate Sintoma
	(slot Dor_no_Peito)
	(slot Dor_no_Braco_Esquerdo)
	(slot Falta_de_Ar)
	(slot Suor_Frio)
	(slot Nausea)
	(slot Desmaio)
    (slot Falta_de_Ar_ao_Deitar)
	(slot Fadiga)
	(slot Ganho_de_Peso)
	(slot Tosse_Noturna)
	(slot Palpitacoes)
	(slot Tontura)
	(slot Ansiedade_Subita)
	(slot Claudicacao)
	(slot Feridas_Que_Nao_Cicatrizam)
	(slot Pele_Fria_ou_Palida)
	(slot Barulho_no_Coracao)
	(slot Inchaço)
	(slot Sangramento_Nasal)
	(slot Visao_Embacada)
	(slot Dor_de_Cabeca)
	(slot Zumbido)
	(slot Dormencia_Formigamento)
	(slot Pulsacao_Abdominal)
	(slot Sinais_de_Choque)
	(slot Dor_Subita_Intensa))

; Perguntas dos sintomas
(defrule ObterSintomaDorNoPeito
	=>
	(printout t crlf "Você sente dor no peito? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Dor_no_Peito ?r))))

(defrule ObterSintomaDorNoBracoEsquerdo
	=>
	(printout t crlf "Você sente dor no braço esquerdo? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Dor_no_Braco_Esquerdo ?r))))

(defrule ObterSintomaFaltaDeAr
	=>
	(printout t crlf "Você sente falta de ar? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Falta_de_Ar ?r))))

(defrule ObterSintomaSuorFrio
	=>
	(printout t crlf "Você sente suor frio? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Suor_Frio ?r))))

(defrule ObterSintomaNausea
	=>
	(printout t crlf "Você sente náuseas? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Nausea ?r))))

(defrule ObterSintomaDesmaio
	=>
	(printout t crlf "Você desmaia ou tem episódios de inconsciência? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Desmaio ?r))))

(defrule ObterSintomaFaltaDeArAoDeitar
	=>
	(printout t crlf "Você sente falta de ar ao deitar? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Falta_de_Ar_ao_Deitar ?r))))

(defrule ObterSintomaFadiga
	=>
	(printout t crlf "Você sente fadiga frequente? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Fadiga ?r))))

(defrule ObterSintomaGanhoDePeso
	=>
	(printout t crlf "Você teve ganho de peso recente? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Ganho_de_Peso ?r))))

(defrule ObterSintomaTosseNoturna
	=>
	(printout t crlf "Você tem tosse noturna? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Tosse_Noturna ?r))))

(defrule ObterSintomaPalpitacoes
	=>
	(printout t crlf "Você sente palpitações? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Palpitacoes ?r))))

(defrule ObterSintomaTontura
	=>
	(printout t crlf "Você sente tontura? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Tontura ?r))))

(defrule ObterSintomaAnsiedadeSubita
	=>
	(printout t crlf "Você sente ansiedade súbita? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Ansiedade_Subita ?r))))

(defrule ObterSintomaClaudicacao
	=>
	(printout t crlf "Você sente dor ao caminhar (claudicação)? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Claudicacao ?r))))

(defrule ObterSintomaFeridas
	=>
	(printout t crlf "Você tem feridas nos pés que não cicatrizam? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Feridas_Que_Nao_Cicatrizam ?r))))

(defrule ObterSintomaPeleFria
	=>
	(printout t crlf "Você sente a pele fria ou pálida nas extremidades? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Pele_Fria_ou_Palida ?r))))

(defrule ObterSintomaBarulhoNoCoracao
	=>
	(printout t crlf "Você sente barulho no coração? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Barulho_no_Coracao ?r))))

(defrule ObterSintomaInchaco
	=>
	(printout t crlf "Você tem inchaços nas pernas ou tornozelos? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Inchaço ?r))))

(defrule ObterSintomaSangramentoNasal
	=>
	(printout t crlf "Você tem sangramentos nasais? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Sangramento_Nasal ?r))))

(defrule ObterSintomaVisaoEmbacada
	=>
	(printout t crlf "Você tem visão embaçada? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Visao_Embacada ?r))))

(defrule ObterSintomaDorDeCabeca
	=>
	(printout t crlf "Você sente dor de cabeça frequente? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Dor_de_Cabeca ?r))))

(defrule ObterSintomaZumbido
	=>
	(printout t crlf "Você escuta zumbidos nos ouvidos? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Zumbido ?r))))

(defrule ObterSintomaDormencia
	=>
	(printout t crlf "Você sente dormência ou formigamento nos membros? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Dormencia_Formigamento ?r))))

(defrule ObterSintomaPulsacao
	=>
	(printout t crlf "Você sente pulsação no abdômen? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Pulsacao_Abdominal ?r))))

(defrule ObterSintomaSinaisChoque
	=>
	(printout t crlf "Você apresenta sinais de choque (palidez, confusão, pulso fraco)? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Sinais_de_Choque ?r))))

(defrule ObterSintomaDorSubita
	=>
	(printout t crlf "Você sentiu dor súbita e intensa? (yes/no)" crlf crlf)
	(bind ?r (read))
	(assert (Sintoma(Dor_Subita_Intensa ?r))))


; Regras de Diagnóstico
(defrule DiagnosticoInfarto
	(and (Sintoma(Dor_no_Peito yes)) (Sintoma(Dor_no_Braco_Esquerdo yes)) (Sintoma(Falta_de_Ar yes)) 
	(Sintoma(Suor_Frio yes)) (Sintoma(Nausea yes)) (Sintoma(Desmaio yes)))
	=>
	(assert (Doenca(doenca infarto)))
	(printout t crlf "Possível doença: Infarto do Miocárdio" crlf))

(defrule DiagnosticoInsuficienciaCardiaca
	(and (Sintoma(Falta_de_Ar_ao_Deitar yes)) (Sintoma(Fadiga yes)) (Sintoma(Inchaço yes)) 
	(Sintoma(Ganho_de_Peso yes)) (Sintoma(Tosse_Noturna yes)))
	=>
	(assert (Doenca(doenca insuficiencia_cardiaca)))
	(printout t crlf "Possível doença: Insuficiência Cardíaca" crlf))

(defrule DiagnosticoArritmia
	(and (Sintoma(Palpitacoes yes)) (Sintoma(Tontura yes)) (Sintoma(Falta_de_Ar yes)) (Sintoma(Ansiedade_Subita yes)))
	=>
	(assert (Doenca(doenca arritmia)))
	(printout t crlf "Possível doença: Arritmia Cardíaca" crlf))

(defrule DiagnosticoDAP
	(and (Sintoma(Claudicacao yes)) (Sintoma(Feridas_Que_Nao_Cicatrizam yes)) (Sintoma(Pele_Fria_ou_Palida yes)))
	=>
	(assert (Doenca(doenca dap)))
	(printout t crlf "Possível doença: Doença Arterial Periférica (DAP)" crlf))

(defrule DiagnosticoValvopatia
	(and (Sintoma(Fadiga yes)) (Sintoma(Tontura yes)) (Sintoma(Desmaio yes)) 
	(Sintoma(Barulho_no_Coracao yes)) (Sintoma(Dor_no_Peito yes)) (Sintoma(Inchaço yes)))
	=>
	(assert (Doenca(doenca valvopatia)))
	(printout t crlf "Possível doença: Valvopatia" crlf))

(defrule DiagnosticoHipertensao
	(and (Sintoma(Sangramento_Nasal yes)) (Sintoma(Visao_Embacada yes)) 
	(Sintoma(Dor_de_Cabeca yes)) (Sintoma(Zumbido yes)) (Sintoma(Tontura yes)))
	=>
	(assert (Doenca(doenca hipertensao)))
	(printout t crlf "Possível doença: Hipertensão Arterial" crlf))

(defrule DiagnosticoAterosclerose
	(and (Sintoma(Dormencia_Formigamento yes)) (Sintoma(Claudicacao yes)) (Sintoma(Dor_no_Peito yes)))
	=>
	(assert (Doenca(doenca aterosclerose)))
	(printout t crlf "Possível doença: Aterosclerose" crlf))

(defrule DiagnosticoAneurisma
	(and (Sintoma(Pulsacao_Abdominal yes)) (Sintoma(Nausea yes)) 
	(Sintoma(Sinais_de_Choque yes)) (Sintoma(Dor_Subita_Intensa yes)))
	=>
	(assert (Doenca(doenca aneurisma)))
	(printout t crlf "Possível doença: Aneurisma" crlf))


; Tratamentos
(defrule TratamentoInfarto
	(Doenca(doenca infarto))
	=>
	(assert (Tratamento(tratamento aspirina_nitroglicerina)))
	(printout t "Tratamento: Aspirina, Nitroglicerina, Trombolíticos e Angioplastia de Emergência" crlf))

(defrule TratamentoInsuficiencia
	(Doenca(doenca insuficiencia_cardiaca))
	=>
	(assert (Tratamento(tratamento diureticos_e_inibidores_ECA)))
	(printout t "Tratamento: Diuréticos, Inibidores de ECA e Betabloqueadores" crlf))

(defrule TratamentoArritmia
	(Doenca(doenca arritmia))
	=>
	(assert (Tratamento(tratamento antiarritmicos)))
	(printout t "Tratamento: Anticoagulantes, Antiarritmicos, Cardioversão ou Ablação por Cateter" crlf))

(defrule TratamentoDAP
	(Doenca(doenca dap))
	=>
	(assert (Tratamento(tratamento antiplaquetarios_estatinas)))
	(printout t "Tratamento: Angioplastia de Emergência, Antiplaquetários, Estatinas, Cirurgia ou Mudança de Estilo de Vida" crlf))

(defrule TratamentoValvopatia
	(Doenca(doenca valvopatia))
	=>
	(assert (Tratamento(tratamento valvular)))
	(printout t "Tratamento: Substituição Valvar, Betabloqueadores, Diuréticos, Anticoagulantes" crlf))

(defrule TratamentoHipertensao
	(Doenca(doenca hipertensao))
	=>
	(assert (Tratamento(tratamento hipertensao)))
	(printout t "Tratamento: Diuréticos, Inibidores de ECA, Mudança de Estilo de Vida" crlf))

(defrule TratamentoAterosclerose
	(Doenca(doenca aterosclerose))
	=>
	(assert (Tratamento(tratamento estatinas_antiplaquetarios)))
	(printout t "Tratamento: Antiplaquetários, Angioplastia ou estatinas" crlf))

(defrule TratamentoAneurisma
	(Doenca(doenca aneurisma))
	=>
	(assert (Tratamento(tratamento aneurisma)))
	(printout t "Tratamento: Betabloqueadores, estatinas e cirurgia" crlf))


; Título
(defrule titulo
	(declare (salience 10))
	=>
	(printout t crlf)
	(printout t "Sistema Especialista - Doenças Cardíacas")
	(printout t crlf))
